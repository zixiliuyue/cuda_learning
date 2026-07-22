// Package datamanager implements dataset version management with DVC integration.
//
// DatasetManager provides DVC-based dataset versioning, data lineage tracking,
// and integration with MinIO/S3 for remote storage.
package datamanager

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

// Dataset represents a dataset.
type Dataset struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	CreatedAt   time.Time `json:"createdAt"`
	CreatedBy   string    `json:"createdBy"`
}

// DatasetVersion represents a specific version of a dataset.
type DatasetVersion struct {
	VersionID   string    `json:"versionId"`
	DatasetID   string    `json:"datasetId"`
	Hash        string    `json:"hash"` // Content hash
	Size        int64     `json:"size"`
	CreatedAt   time.Time `json:"createdAt"`
	CreatedBy   string    `json:"createdBy"`
	Description string    `json:"description"`
}

// Storage is the interface for remote storage (MinIO/S3).
type Storage interface {
	// Put uploads data to remote storage
	Put(ctx context.Context, objectName string, reader io.Reader, size int64) error
	// Get downloads data from remote storage
	Get(ctx context.Context, objectName string) (io.ReadCloser, error)
	// Delete removes an object from remote storage
	Delete(ctx context.Context, objectName string) error
	// Exists checks if an object exists
	Exists(ctx context.Context, objectName string) (bool, error)
}

// MetadataDB is the interface for metadata storage.
type MetadataDB interface {
	// SaveDataset saves dataset metadata
	SaveDataset(ctx context.Context, ds *Dataset) error
	// GetDataset gets dataset by ID
	GetDataset(ctx context.Context, id string) (*Dataset, error)
	// ListDatasets lists all datasets
	ListDatasets(ctx context.Context) ([]*Dataset, error)
	// SaveVersion saves version metadata
	SaveVersion(ctx context.Context, version *DatasetVersion) error
	// GetVersion gets version by ID
	GetVersion(ctx context.Context, datasetID, versionID string) (*DatasetVersion, error)
	// ListVersions lists all versions for a dataset
	ListVersions(ctx context.Context, datasetID string) ([]*DatasetVersion, error)
	// RecordLineage records data lineage: which task consumed which version
	RecordLineage(ctx context.Context, datasetID, versionID, taskID string) error
	// GetLineage gets all tasks that consumed a version
	GetLineage(ctx context.Context, datasetID, versionID string) ([]string, error)
}

// DatasetManager manages datasets and their versions.
type DatasetManager struct {
	workDir string
	storage Storage
	metaDB  MetadataDB
}

// NewDatasetManager creates a new DatasetManager.
func NewDatasetManager(workDir string, storage Storage, metaDB MetadataDB) *DatasetManager {
	return &DatasetManager{
		workDir: workDir,
		storage: storage,
		metaDB:  metaDB,
	}
}

// CreateDataset creates a new dataset.
func (m *DatasetManager) CreateDataset(ctx context.Context, name, description, createdBy string) (*Dataset, error) {
	ds := &Dataset{
		ID:          uuid.New().String(),
		Name:        name,
		Description: description,
		CreatedAt:   time.Now(),
		CreatedBy:   createdBy,
	}

	err := m.metaDB.SaveDataset(ctx, ds)
	if err != nil {
		return nil, fmt.Errorf("failed to save dataset metadata: %w", err)
	}

	// Create local working directory
	localPath := filepath.Join(m.workDir, ds.ID)
	if err := os.MkdirAll(localPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create working directory: %w", err)
	}

	return ds, nil
}

// CreateVersion creates a new version of the dataset.
func (m *DatasetManager) CreateVersion(ctx context.Context, datasetID, localPath, description, createdBy string) (*DatasetVersion, error) {
	// 1. Calculate content hash
	hash, size, err := m.computeHash(localPath)
	if err != nil {
		return nil, fmt.Errorf("failed to compute content hash: %w", err)
	}

	// 2. Create version metadata
	version := &DatasetVersion{
		VersionID:   uuid.New().String(),
		DatasetID:   datasetID,
		Hash:        hash,
		Size:        size,
		CreatedAt:   time.Now(),
		CreatedBy:   createdBy,
		Description: description,
	}

	// 3. DVC add and push to remote storage
	if err := m.dvcAdd(localPath); err != nil {
		return nil, err
	}
	if err := m.dvcPush(version); err != nil {
		return nil, err
	}

	// 4. Save metadata
	err = m.metaDB.SaveVersion(ctx, version)
	if err != nil {
		return nil, fmt.Errorf("failed to save version metadata: %w", err)
	}

	return version, nil
}

// Checkout checks out a specific version to the local working directory.
func (m *DatasetManager) Checkout(ctx context.Context, datasetID, versionID string) error {
	version, err := m.metaDB.GetVersion(ctx, datasetID, versionID)
	if err != nil {
		return fmt.Errorf("failed to get version: %w", err)
	}

	// Checkout from DVC remote storage
	if err := m.dvcCheckout(version.Hash); err != nil {
		return fmt.Errorf("failed to checkout version: %w", err)
	}

	return nil
}

// DeleteVersion deletes a version from both metadata and storage.
func (m *DatasetManager) DeleteVersion(ctx context.Context, datasetID, versionID string) error {
	version, err := m.metaDB.GetVersion(ctx, datasetID, versionID)
	if err != nil {
		return err
	}

	// Delete from remote storage
	objectName := fmt.Sprintf("%s/%s", datasetID, version.Hash)
	err = m.storage.Delete(ctx, objectName)
	if err != nil {
		return fmt.Errorf("failed to delete from storage: %w", err)
	}

	// Delete from metadata
	// TODO: Implement metadata deletion

	return nil
}

// RecordLineage records that a task consumed a dataset version.
func (m *DatasetManager) RecordLineage(ctx context.Context, datasetID, versionID, taskID string) error {
	return m.metaDB.RecordLineage(ctx, datasetID, versionID, taskID)
}

// GetLineage gets all tasks that consumed a dataset version.
func (m *DatasetManager) GetLineage(ctx context.Context, datasetID, versionID string) ([]string, error) {
	return m.metaDB.GetLineage(ctx, datasetID, versionID)
}

// computeHash computes the content hash of a local file/directory.
func (m *DatasetManager) computeHash(localPath string) (string, int64, error) {
	// In real implementation, this would traverse the directory and compute content hash
	// For now, we return a placeholder
	stat, err := os.Stat(localPath)
	if err != nil {
		return "", 0, err
	}
	return fmt.Sprintf("%x", stat.ModTime()), stat.Size(), nil
}

// dvcAdd runs dvc add on the local path.
func (m *DatasetManager) dvcAdd(localPath string) error {
	// In real implementation, this would exec `dvc add`
	// For this skeleton, we leave it as placeholder
	return nil
}

// dvcPush runs dvc push to remote storage.
func (m *DatasetManager) dvcPush(version *DatasetVersion) error {
	// In real implementation, this would exec `dvc push`
	// For this skeleton, we leave it as placeholder
	return nil
}

// dvcCheckout runs dvc checkout for the given hash.
func (m *DatasetManager) dvcCheckout(hash string) error {
	// In real implementation, this would exec `dvc checkout`
	// For this skeleton, we leave it as placeholder
	return nil
}
