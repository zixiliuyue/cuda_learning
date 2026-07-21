// Package auth 实现平台的认证（JWT）与授权（RBAC）能力。
//
// This file provides OIDC OAuth 2.0 SSO 企业单点登录支持。
package auth

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/coreos/go-oidc/v3/oidc"
	"golang.org/x/oauth2"

	"qingtian/pkg/apperr"
)

// OIDCConfig holds the OIDC provider configuration.
type OIDCConfig struct {
	Issuer       string
	ClientID     string
	ClientSecret string
	RedirectURL  string
	Scopes       string
	UserNameClaim string
}

// OIDCAuthenticator handles OIDC OAuth 2.0 authentication flow.
//
// It:
//   - Discovers the OIDC provider configuration
//   - Verifies ID tokens issued by the provider
//   - Provides OAuth 2.0 flow for login redirect and callback
type OIDCAuthenticator struct {
	config       OIDCConfig
	provider     *oidc.Provider
	verifier     *oidc.IDTokenVerifier
	oauthConfig  *oauth2.Config
}

// NewOIDCAuthenticator creates and initializes a new OIDC authenticator.
//
// It performs OIDC discovery to get provider metadata and keys.
func NewOIDCAuthenticator(ctx context.Context, cfg OIDCConfig) (*OIDCAuthenticator, error) {
	provider, err := oidc.NewProvider(ctx, cfg.Issuer)
	if err != nil {
		return nil, apperr.Wrap(apperr.CodeInternal, "OIDC provider discovery failed", err)
	}

	config := &oidc.Config{
		ClientID: cfg.ClientID,
	}
	verifier := provider.Verifier(config)

	scopes := parseScopes(cfg.Scopes)
	oauthConfig := &oauth2.Config{
		ClientID:     cfg.ClientID,
		ClientSecret: cfg.ClientSecret,
		RedirectURL:  cfg.RedirectURL,
		Endpoint:     provider.Endpoint(),
		Scopes:       scopes,
	}

	return &OIDCAuthenticator{
		config:      cfg,
		provider:    provider,
		verifier:    verifier,
		oauthConfig: oauthConfig,
	}, nil
}

// GetAuthURL returns the URL to redirect the user to for OIDC authentication.
//
// The state is used to prevent CSRF and should be stored and validated in the callback.
func (a *OIDCAuthenticator) GetAuthURL(state string) string {
	return a.oauthConfig.AuthCodeURL(state)
}

// ExchangeCode exchanges an authorization code for an access token and ID token.
//
// Returns the verified ID token claims and the raw tokens on success.
func (a *OIDCAuthenticator) ExchangeCode(ctx context.Context, code string) (*oidc.IDToken, *oauth2.Token, error) {
	token, err := a.oauthConfig.Exchange(ctx, code)
	if err != nil {
		return nil, nil, apperr.Wrap(apperr.CodeInternal, "failed to exchange authorization code", err)
	}

	rawIDToken, ok := token.Extra("id_token").(string)
	if !ok {
		return nil, nil, apperr.InvalidArgument("id_token not found in token response")
	}

	idToken, err := a.verifier.Verify(ctx, rawIDToken)
	if err != nil {
		return nil, nil, apperr.Wrap(apperr.CodeUnauthorized, "failed to verify id_token", err)
	}

	return idToken, token, nil
}

// ExtractClaims extracts claims from the ID token into the provided struct.
//
// Example usage:
//
//	var claims struct {
//		Email string `json:"email"`
//		Sub   string `json:"sub"`
//	}
//	if err := authenticator.ExtractClaims(idToken, &claims); err != nil { ... }
func (a *OIDCAuthenticator) ExtractClaims(idToken *oidc.IDToken, claims interface{}) error {
	return idToken.Claims(claims)
}

// GetUserNameClaim returns the configured username claim name.
func (a *OIDCAuthenticator) GetUserNameClaim() string {
	if a.config.UserNameClaim == "" {
		return "email"
	}
	return a.config.UserNameClaim
}

// UserInfo contains the user information extracted from OIDC claims.
type UserInfo struct {
	Subject     string // sub - external user ID from OIDC provider
	Username    string // username/email from configured claim
	Email       string // email claim
	Name        string // name claim (full name)
	IsEmailVerified bool // email_verified claim
}

// ExtractUserInfo extracts common user claims from the ID token.
//
// This is a convenience helper that extracts commonly used claims.
func (a *OIDCAuthenticator) ExtractUserInfo(idToken *oidc.IDToken) (*UserInfo, error) {
	var claims struct {
		Sub           string `json:"sub"`
		Email         string `json:"email"`
		Name          string `json:"name"`
		EmailVerified bool   `json:"email_verified"`
	}
	if err := a.ExtractClaims(idToken, &claims); err != nil {
		return nil, err
	}

	usernameClaim := a.GetUserNameClaim()
	var username string
	if usernameClaim == "email" {
		username = claims.Email
	} else {
		// Try to extract custom username claim
		var customClaims map[string]interface{}
		if err := a.ExtractClaims(idToken, &customClaims); err == nil {
			if val, ok := customClaims[usernameClaim]; ok {
				if s, ok := val.(string); ok {
					username = s
				}
			}
		}
		// Fallback to email if username claim not found
		if username == "" {
			username = claims.Email
		}
	}

	return &UserInfo{
		Subject:         claims.Sub,
		Username:        username,
		Email:           claims.Email,
		Name:            claims.Name,
		IsEmailVerified: claims.EmailVerified,
	}, nil
}

// OIDCUserInfo returns user info from the userinfo endpoint.
//
// Some OIDC providers require additional calls to get user info.
func (a *OIDCAuthenticator) OIDCUserInfo(ctx context.Context, token *oauth2.Token) (map[string]interface{}, error) {
	uiEndpoint := a.provider.UserInfoEndpoint()
	if uiEndpoint == "" {
		return nil, nil
	}

	req, err := http.NewRequestWithContext(ctx, "GET", uiEndpoint, nil)
	if err != nil {
		return nil, err
	}

	client := oauth2.NewClient(ctx, a.oauthConfig.TokenSource(ctx, token))
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var claims map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&claims); err != nil {
		return nil, err
	}

	return claims, nil
}

// parseScopes splits the scopes string into a slice.
func parseScopes(s string) []string {
	if s == "" {
		return []string{"openid", "email", "profile"}
	}
	// Simple split on whitespace - handles most cases
	var scopes []string
	current := ""
	for _, c := range s {
		if c == ' ' || c == ',' {
			if current != "" {
				scopes = append(scopes, current)
				current = ""
			}
		} else {
			current += string(c)
		}
	}
	if current != "" {
		scopes = append(scopes, current)
	}
	// Ensure openid is always present (required by OIDC spec)
	hasOpenID := false
	for _, scope := range scopes {
		if scope == "openid" {
			hasOpenID = true
			break
		}
	}
	if !hasOpenID {
		scopes = append([]string{"openid"}, scopes...)
	}
	return scopes
}
