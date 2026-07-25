// 本文件实现基于角色的访问控制（RBAC）。
//
// 平台将操作抽象为"权限"，将角色映射到一组权限，
// 通过 HasPermission 判断某角色是否具备执行某操作的权限。
package auth

import "qingtian/internal/domain"

// Permission 表示一项可授权的操作。
type Permission string

const (
	PermTaskCreate  Permission = "task:create"  // 创建任务
	PermTaskView    Permission = "task:view"     // 查看任务
	PermTaskCancel  Permission = "task:cancel"   // 取消任务
	PermNodeView    Permission = "node:view"     // 查看节点
	PermNodeManage  Permission = "node:manage"   // 管理节点（如下线）
	PermNodeInstall Permission = "node:install"  // 安装/纳管节点（SSH 一键安装）
	PermUserManage  Permission = "user:manage"   // 管理用户

	PermTemplateView   Permission = "template:view"   // 查看任务模板
	PermTemplateCreate Permission = "template:create" // 创建/修改任务模板
	PermProjectView    Permission = "project:view"    // 查看项目
	PermProjectCreate  Permission = "project:create"  // 创建项目/管理成员
	PermQuotaView      Permission = "quota:view"      // 查看配额
	PermQuotaEdit      Permission = "quota:edit"      // 修改配额
	PermMeteringView   Permission = "metering:view"   // 查看计量计费
	PermWorkflowView   Permission = "workflow:view"   // 查看工作流/调度
	PermWorkflowCreate Permission = "workflow:create" // 创建/修改工作流/调度
)

// rolePermissions 定义每个角色拥有的权限集合。
//
// 设计说明：
//   - admin:     拥有全部权限。
//   - developer: 可创建/查看/取消任务、工作流、调度、模板、项目，查看节点/配额/计量。
//   - viewer:    仅可查看类权限。
var rolePermissions = map[domain.Role]map[Permission]bool{
	domain.RoleAdmin: {
		PermTaskCreate: true, PermTaskView: true, PermTaskCancel: true,
		PermNodeView: true, PermNodeManage: true, PermNodeInstall: true, PermUserManage: true,
		PermTemplateView: true, PermTemplateCreate: true,
		PermProjectView: true, PermProjectCreate: true,
		PermQuotaView: true, PermQuotaEdit: true,
		PermMeteringView: true,
		PermWorkflowView: true, PermWorkflowCreate: true,
	},
	domain.RoleDeveloper: {
		PermTaskCreate: true, PermTaskView: true, PermTaskCancel: true,
		PermNodeView: true,
		PermTemplateView: true, PermTemplateCreate: true,
		PermProjectView: true, PermProjectCreate: true,
		PermQuotaView: true,
		PermMeteringView: true,
		PermWorkflowView: true, PermWorkflowCreate: true,
	},
	domain.RoleViewer: {
		PermTaskView: true, PermNodeView: true,
		PermTemplateView: true, PermProjectView: true,
		PermQuotaView: true, PermMeteringView: true,
		PermWorkflowView: true,
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
