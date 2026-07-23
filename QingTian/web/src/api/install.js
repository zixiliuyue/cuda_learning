// 节点安装 API：一键 SSH 安装 Agent
import request from './request'

// 测试 SSH 连接
export function testSSHConnection(params) {
  return request.post('/nodes/install/test-ssh', params)
}

// 执行安装
export function runInstall(params) {
  return request.post('/nodes/install/run', params)
}

export default {
  testSSH: testSSHConnection,
  installRun: runInstall,
}
