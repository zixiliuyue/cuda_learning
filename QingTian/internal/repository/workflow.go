// Package repository 定义工作流相关的数据访问接口。
//
// 遵循依赖倒置原则：上层服务依赖接口，不依赖具体实现，
// 便于后续替换存储实现（MongoDB/PostgreSQL/...）。
package repository

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"qingtian/internal/domain"
)

// WorkflowRepository 工作流定义存储接口
type WorkflowRepository interface {
	// GetByID 根据 ID 获取工作流定义
	GetByID(ctx context.Context, id string) (*domain.Workflow, error)

	// List 列出工作流定义（分页）
	List(ctx context.Context, createdBy string, skip, limit int) ([]*domain.Workflow, int64, error)

	// Create 创建工作流定义
	Create(ctx context.Context, wf *domain.Workflow) error

	// Update 更新工作流定义
	Update(ctx context.Context, wf *domain.Workflow) error

	// Delete 删除工作流定义（软删除，改为 archived 状态）
	Delete(ctx context.Context, id string) error
}

// WorkflowRunRepository 工作流执行实例存储接口
type WorkflowRunRepository interface {
	// GetByID 根据 ID 获取工作流执行实例
	GetByID(ctx context.Context, id string) (*domain.WorkflowRun, error)

	// Create 创建新的执行实例
	Create(ctx context.Context, run *domain.WorkflowRun) error

	// Update 更新执行实例状态
	Update(ctx context.Context, run *domain.WorkflowRun) error

	// ListByWorkflow 列出工作流的执行历史
	ListByWorkflow(ctx context.Context, workflowID string, skip, limit int) ([]*domain.WorkflowRun, int64, error)
}

// TaskRunRepository 任务执行实例存储接口
type TaskRunRepository interface {
	// GetByID 获取任务执行实例
	GetByID(ctx context.Context, id string) (*domain.TaskRun, error)

	// Create 创建任务执行实例
	Create(ctx context.Context, tr *domain.TaskRun) error

	// Update 更新任务执行实例
	Update(ctx context.Context, tr *domain.TaskRun) error
}

// ScheduleRepository 定时计划存储接口
type ScheduleRepository interface {
	// GetByID 获取定时计划
	GetByID(ctx context.Context, id string) (*domain.Schedule, error)

	// List 列出所有定时计划
	List(ctx context.Context, skip, limit int) ([]*domain.Schedule, int64, error)

	// Create 创建定时计划
	Create(ctx context.Context, s *domain.Schedule) error

	// Update 更新定时计划
	Update(ctx context.Context, s *domain.Schedule) error

	// Delete 删除定时计划
	Delete(ctx context.Context, id string) error

	// UpdateNextRunTime 更新下次执行时间
	UpdateNextRunTime(ctx context.Context, id string, nextRun *time.Time) error

	// ListToRun 列出所有需要在这个时间窗口内执行的计划
	ListToRun(ctx context.Context, now time.Time, limit int) ([]*domain.Schedule, error)
}

// --- MongoDB 具体实现 ---

// mongoWorkflowRepo MongoDB 实现工作流存储
type mongoWorkflowRepo struct {
	coll *mongo.Collection
}

// NewMongoWorkflowRepo 创建 MongoDB 工作流存储实现
func NewMongoWorkflowRepo(db *mongo.Database) WorkflowRepository {
	return &mongoWorkflowRepo{
		coll: db.Collection("workflows"),
	}
}

// GetByID implements WorkflowRepository
func (r *mongoWorkflowRepo) GetByID(ctx context.Context, id string) (*domain.Workflow, error) {
	var wf domain.Workflow
	filter := bson.D{{Key: "_id", Value: id}}
	err := r.coll.FindOne(ctx, filter).Decode(&wf)
	if err != nil {
		return nil, err
	}
	return &wf, nil
}

// List implements WorkflowRepository
func (r *mongoWorkflowRepo) List(ctx context.Context, createdBy string, skip, limit int) ([]*domain.Workflow, int64, error) {
	filter := bson.D{}
	if createdBy != "" {
		filter = append(filter, bson.E{Key: "createdBy", Value: createdBy})
	}
	var wfs []*domain.Workflow
	findOpts := options.Find().SetSkip(int64(skip)).SetLimit(int64(limit))
	cursor, err := r.coll.Find(ctx, filter, findOpts)
	if err != nil {
		return nil, 0, err
	}
	err = cursor.All(ctx, &wfs)
	if err != nil {
		return nil, 0, err
	}
	total, err := r.coll.CountDocuments(ctx, filter)
	if err != nil {
		return nil, 0, err
	}
	return wfs, total, nil
}

// Create implements WorkflowRepository
func (r *mongoWorkflowRepo) Create(ctx context.Context, wf *domain.Workflow) error {
	_, err := r.coll.InsertOne(ctx, wf)
	return err
}

// Update implements WorkflowRepository
func (r *mongoWorkflowRepo) Update(ctx context.Context, wf *domain.Workflow) error {
	filter := bson.D{{Key: "_id", Value: wf.ID}}
	_, err := r.coll.ReplaceOne(ctx, filter, wf)
	return err
}

// Delete implements WorkflowRepository
func (r *mongoWorkflowRepo) Delete(ctx context.Context, id string) error {
	// 软删除：改为 archived 状态
	update := bson.D{{Key: "status", Value: domain.WorkflowStatusArchived}}
	filter := bson.D{{Key: "_id", Value: id}}
	_, err := r.coll.UpdateOne(ctx, filter, bson.D{{Key: "$set", Value: update}})
	return err
}

// mongoWorkflowRunRepo MongoDB 实现工作流执行实例存储
type mongoWorkflowRunRepo struct {
	coll *mongo.Collection
}

// NewMongoWorkflowRunRepo 创建 MongoDB 工作流执行实例存储
func NewMongoWorkflowRunRepo(db *mongo.Database) WorkflowRunRepository {
	return &mongoWorkflowRunRepo{
		coll: db.Collection("workflow_runs"),
	}
}

// GetByID implements WorkflowRunRepository
func (r *mongoWorkflowRunRepo) GetByID(ctx context.Context, id string) (*domain.WorkflowRun, error) {
	var run domain.WorkflowRun
	filter := bson.D{{Key: "_id", Value: id}}
	err := r.coll.FindOne(ctx, filter).Decode(&run)
	if err != nil {
		return nil, err
	}
	return &run, nil
}

// Create implements WorkflowRunRepository
func (r *mongoWorkflowRunRepo) Create(ctx context.Context, run *domain.WorkflowRun) error {
	_, err := r.coll.InsertOne(ctx, run)
	return err
}

// Update implements WorkflowRunRepository
func (r *mongoWorkflowRunRepo) Update(ctx context.Context, run *domain.WorkflowRun) error {
	filter := bson.D{{Key: "_id", Value: run.ID}}
	_, err := r.coll.ReplaceOne(ctx, filter, run)
	return err
}

// ListByWorkflow implements WorkflowRunRepository
func (r *mongoWorkflowRunRepo) ListByWorkflow(ctx context.Context, workflowID string, skip, limit int) ([]*domain.WorkflowRun, int64, error) {
	filter := bson.D{{Key: "workflowId", Value: workflowID}}
	var runs []*domain.WorkflowRun
	findOpts := options.Find().SetSkip(int64(skip)).SetLimit(int64(limit))
	cursor, err := r.coll.Find(ctx, filter, findOpts)
	if err != nil {
		return nil, 0, err
	}
	err = cursor.All(ctx, &runs)
	if err != nil {
		return nil, 0, err
	}
	total, err := r.coll.CountDocuments(ctx, filter)
	if err != nil {
		return nil, 0, err
	}
	return runs, total, nil
}

// mongoTaskRunRepo MongoDB 实现任务执行实例存储
type mongoTaskRunRepo struct {
	coll *mongo.Collection
}

// NewMongoTaskRunRepo 创建 MongoDB 任务执行实例存储
func NewMongoTaskRunRepo(db *mongo.Database) TaskRunRepository {
	return &mongoTaskRunRepo{
		coll: db.Collection("task_runs"),
	}
}

// GetByID implements TaskRunRepository
func (r *mongoTaskRunRepo) GetByID(ctx context.Context, id string) (*domain.TaskRun, error) {
	var tr domain.TaskRun
	filter := bson.D{{Key: "_id", Value: id}}
	err := r.coll.FindOne(ctx, filter).Decode(&tr)
	if err != nil {
		return nil, err
	}
	return &tr, nil
}

// Create implements TaskRunRepository
func (r *mongoTaskRunRepo) Create(ctx context.Context, tr *domain.TaskRun) error {
	_, err := r.coll.InsertOne(ctx, tr)
	return err
}

// Update implements TaskRunRepository
func (r *mongoTaskRunRepo) Update(ctx context.Context, tr *domain.TaskRun) error {
	filter := bson.D{{Key: "_id", Value: tr.ID}}
	_, err := r.coll.ReplaceOne(ctx, filter, tr)
	return err
}

// mongoScheduleRepo MongoDB 实现定时计划存储
type mongoScheduleRepo struct {
	coll *mongo.Collection
}

// NewMongoScheduleRepo 创建 MongoDB 定时计划存储
func NewMongoScheduleRepo(db *mongo.Database) ScheduleRepository {
	return &mongoScheduleRepo{
		coll: db.Collection("schedules"),
	}
}

// GetByID implements ScheduleRepository
func (r *mongoScheduleRepo) GetByID(ctx context.Context, id string) (*domain.Schedule, error) {
	var s domain.Schedule
	filter := bson.D{{Key: "_id", Value: id}}
	err := r.coll.FindOne(ctx, filter).Decode(&s)
	if err != nil {
		return nil, err
	}
	return &s, nil
}

// List implements ScheduleRepository
func (r *mongoScheduleRepo) List(ctx context.Context, skip, limit int) ([]*domain.Schedule, int64, error) {
	var schedules []*domain.Schedule
	findOpts := options.Find().SetSkip(int64(skip)).SetLimit(int64(limit))
	cursor, err := r.coll.Find(ctx, findOpts)
	if err != nil {
		return nil, 0, err
	}
	err = cursor.All(ctx, &schedules)
	if err != nil {
		return nil, 0, err
	}
	total, err := r.coll.CountDocuments(ctx, bson.D{})
	if err != nil {
		return nil, 0, err
	}
	return schedules, total, nil
}

// Create implements ScheduleRepository
func (r *mongoScheduleRepo) Create(ctx context.Context, s *domain.Schedule) error {
	_, err := r.coll.InsertOne(ctx, s)
	return err
}

// Update implements ScheduleRepository
func (r *mongoScheduleRepo) Update(ctx context.Context, s *domain.Schedule) error {
	filter := bson.D{{Key: "_id", Value: s.ID}}
	_, err := r.coll.ReplaceOne(ctx, filter, s)
	return err
}

// Delete implements ScheduleRepository
func (r *mongoScheduleRepo) Delete(ctx context.Context, id string) error {
	_, err := r.coll.DeleteOne(ctx, bson.D{{Key: "_id", Value: id}})
	return err
}

// UpdateNextRunTime implements ScheduleRepository
func (r *mongoScheduleRepo) UpdateNextRunTime(ctx context.Context, id string, nextRun *time.Time) error {
	filter := bson.D{{Key: "_id", Value: id}}
	update := bson.D{
		{Key: "nextRunAt", Value: nextRun},
		{Key: "updatedAt", Value: time.Now()},
	}
	_, err := r.coll.UpdateOne(ctx, filter, update)
	return err
}

// ListToRun implements ScheduleRepository
func (r *mongoScheduleRepo) ListToRun(ctx context.Context, now time.Time, limit int) ([]*domain.Schedule, error) {
	// 找出所有 enabled 且 nextRunAt <= now 的计划
	filter := bson.D{
		{Key: "enabled", Value: true},
		{Key: "nextRunAt", Value: bson.D{{Key: "$lte", Value: now}}},
	}
	findOpts := options.Find().SetLimit(int64(limit))
	cursor, err := r.coll.Find(ctx, filter, findOpts)
	if err != nil {
		return nil, err
	}
	var schedules []*domain.Schedule
	err = cursor.All(ctx, &schedules)
	if err != nil {
		return nil, err
	}
	return schedules, err
}
