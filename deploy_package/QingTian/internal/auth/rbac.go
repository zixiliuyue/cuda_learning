// 本文件实现基于角色的访问控制（RBAC）。
//
// 平台将操作抽象为"权限"，将角色映射到一组权限，
// 通过 HasPermission 判断某角色是否具备执行某操作的权限。
package auth

import "qingtian/internal/domain"

// Permission 表示一项可授权的操作。
type Permission string

const (
	PermTaskCreate Permission = "task:create" // 创建任务
	PermTaskView   Permission = "task:view"   // 查看任务
	PermTaskCancel Permission = "task:cancel" // 取消任务
	PermNodeView   Permission = "node:view"   // 查看节点
	PermNodeManage Permission = "node:manage" // 管理节点（如下线）
	PermUserManage Permission = "user:manage" // 管理用户
)

// rolePermissions 定义每个角色拥有的权限集合。
//
// 设计说明：
//   - admin:     拥有全部权限。
//   - developer: 可创建/查看/取消任务，查看节点。
//   - viewer:    仅可查看任务与节点。
var rolePermissions = map[domain.Role]map[Permission]bool{
	domain.RoleAdmin: {
		PermTaskCreate: true, PermTaskView: true, PermTaskCancel: true,
		PermNodeView: true, PermNodeManage: true, PermUserManage: true,
	},
	domain.RoleDeveloper: {
		PermTaskCreate: true, PermTaskView: true, PermTaskCancel: true,
		PermNodeView: true,
	},
	domain.RoleViewer: {
		PermTaskView: true, PermNodeView: true,
	},
}

// HasPermission 判断指定角色是否具备指定权限。
//
// 参数：
//   - role: 用户角色。
//   - perm: 待判断的权限。
//
// 返回值：
//   - bool: true 表示具备该权限。
func HasPermission(role domain.Role, perm Permission) bool {
	perms, ok := rolePermissions[role]
	if !ok {
		return false
	}
	return perms[perm]
}
