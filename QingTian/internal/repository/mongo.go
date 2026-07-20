// 本文件提供基于 MongoDB 的仓储实现。
//
// 生产环境使用该实现持久化用户、节点、任务等元数据。
// 每类实体对应一个集合（collection），通过 _id 作为主键。
package repository

import (
	"context"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"qingtian/internal/domain"
	"qingtian/pkg/apperr"
)

// MongoStore 聚合基于 MongoDB 的各仓储实现，持有数据库句柄。
type MongoStore struct {
	db *mongo.Database
}

// NewMongoStore 封装一个 mongo 数据库句柄。
//
// 参数：
//   - db: 已连接的 *mongo.Database。
func NewMongoStore(db *mongo.Database) *MongoStore {
	return &MongoStore{db: db}
}

// EnsureIndexes 创建集合的必要索引，应在服务启动时调用一次。
//
// 目前保证 users.username 唯一，作为并发注册下防止重复账号的最后防线。
// 服务层已有查重逻辑，此处是数据库层兜底；两者配合避免重复用户名。
//
// 注意：若集合中已存在重复 username 数据，创建唯一索引会失败，
// 需先清理重复数据（见部署脚本 --dedup-users）。
func (s *MongoStore) EnsureIndexes(ctx context.Context) error {
	userIdx := mongo.IndexModel{
		Keys:    bson.D{{Key: "username", Value: 1}},
		Options: options.Index().SetUnique(true).SetName("uniq_username"),
	}
	_, err := s.db.Collection("users").Indexes().CreateOne(ctx, userIdx)
	return err
}

// Users 返回用户仓储的 MongoDB 实现。
func (s *MongoStore) Users() *MongoUserRepo { return &MongoUserRepo{col: s.db.Collection("users")} }

// Nodes 返回节点仓储的 MongoDB 实现。
func (s *MongoStore) Nodes() *MongoNodeRepo { return &MongoNodeRepo{col: s.db.Collection("nodes")} }

// Tasks 返回任务仓储的 MongoDB 实现。
func (s *MongoStore) Tasks() *MongoTaskRepo { return &MongoTaskRepo{col: s.db.Collection("tasks")} }

// Usages 返回用量仓储的 MongoDB 实现。
func (s *MongoStore) Usages() *MongoUsageRepo {
	return &MongoUsageRepo{col: s.db.Collection("usages")}
}

// Quotas 返回配额仓储的 MongoDB 实现。
func (s *MongoStore) Quotas() *MongoQuotaRepo {
	return &MongoQuotaRepo{col: s.db.Collection("quotas")}
}

// Projects 返回项目仓储的 MongoDB 实现。
func (s *MongoStore) Projects() *MongoProjectRepo {
	return &MongoProjectRepo{col: s.db.Collection("projects")}
}

// MongoUserRepo 是 UserRepository 的 MongoDB 实现。
type MongoUserRepo struct{ col *mongo.Collection }

// Create 实现 UserRepository.Create。
func (r *MongoUserRepo) Create(ctx context.Context, u *domain.User) error {
	_, err := r.col.InsertOne(ctx, u)
	if mongo.IsDuplicateKeyError(err) {
		return apperr.Conflict("用户名已存在: " + u.Username)
	}
	return err
}

// GetByID 实现 UserRepository.GetByID。
func (r *MongoUserRepo) GetByID(ctx context.Context, id string) (*domain.User, error) {
	var u domain.User
	err := r.col.FindOne(ctx, bson.M{"_id": id}).Decode(&u)
	if err == mongo.ErrNoDocuments {
		return nil, apperr.NotFound("用户不存在: " + id)
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// GetByUsername 实现 UserRepository.GetByUsername。
func (r *MongoUserRepo) GetByUsername(ctx context.Context, username string) (*domain.User, error) {
	var u domain.User
	err := r.col.FindOne(ctx, bson.M{"username": username}).Decode(&u)
	if err == mongo.ErrNoDocuments {
		return nil, apperr.NotFound("用户不存在: " + username)
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// List 实现 UserRepository.List。
func (r *MongoUserRepo) List(ctx context.Context) ([]*domain.User, error) {
	cur, err := r.col.Find(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []*domain.User
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// MongoNodeRepo 是 NodeRepository 的 MongoDB 实现。
type MongoNodeRepo struct{ col *mongo.Collection }

// Upsert 实现 NodeRepository.Upsert。
func (r *MongoNodeRepo) Upsert(ctx context.Context, n *domain.Node) error {
	opts := options.Replace().SetUpsert(true)
	_, err := r.col.ReplaceOne(ctx, bson.M{"_id": n.ID}, n, opts)
	return err
}

// GetByID 实现 NodeRepository.GetByID。
func (r *MongoNodeRepo) GetByID(ctx context.Context, id string) (*domain.Node, error) {
	var n domain.Node
	err := r.col.FindOne(ctx, bson.M{"_id": id}).Decode(&n)
	if err == mongo.ErrNoDocuments {
		return nil, apperr.NotFound("节点不存在: " + id)
	}
	if err != nil {
		return nil, err
	}
	return &n, nil
}

// List 实现 NodeRepository.List。
func (r *MongoNodeRepo) List(ctx context.Context) ([]*domain.Node, error) {
	cur, err := r.col.Find(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []*domain.Node
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// UpdateStatus 实现 NodeRepository.UpdateStatus。
func (r *MongoNodeRepo) UpdateStatus(ctx context.Context, id string, status domain.NodeStatus) error {
	res, err := r.col.UpdateOne(ctx, bson.M{"_id": id}, bson.M{"$set": bson.M{"status": status}})
	if err != nil {
		return err
	}
	if res.MatchedCount == 0 {
		return apperr.NotFound("节点不存在: " + id)
	}
	return nil
}

// UpdateHeartbeat 实现 NodeRepository.UpdateHeartbeat。
func (r *MongoNodeRepo) UpdateHeartbeat(ctx context.Context, id string, gpus []domain.GPU) error {
	set := bson.M{"status": domain.NodeOnline}
	if gpus != nil {
		set["gpus"] = gpus
	}
	res, err := r.col.UpdateOne(ctx, bson.M{"_id": id}, bson.M{"$set": set})
	if err != nil {
		return err
	}
	if res.MatchedCount == 0 {
		return apperr.NotFound("节点不存在: " + id)
	}
	return nil
}

// MongoTaskRepo 是 TaskRepository 的 MongoDB 实现。
type MongoTaskRepo struct{ col *mongo.Collection }

// Create 实现 TaskRepository.Create。
func (r *MongoTaskRepo) Create(ctx context.Context, t *domain.Task) error {
	_, err := r.col.InsertOne(ctx, t)
	if mongo.IsDuplicateKeyError(err) {
		return apperr.Conflict("任务已存在: " + t.ID)
	}
	return err
}

// GetByID 实现 TaskRepository.GetByID。
func (r *MongoTaskRepo) GetByID(ctx context.Context, id string) (*domain.Task, error) {
	var t domain.Task
	err := r.col.FindOne(ctx, bson.M{"_id": id}).Decode(&t)
	if err == mongo.ErrNoDocuments {
		return nil, apperr.NotFound("任务不存在: " + id)
	}
	if err != nil {
		return nil, err
	}
	return &t, nil
}

// Update 实现 TaskRepository.Update。
func (r *MongoTaskRepo) Update(ctx context.Context, t *domain.Task) error {
	res, err := r.col.ReplaceOne(ctx, bson.M{"_id": t.ID}, t)
	if err != nil {
		return err
	}
	if res.MatchedCount == 0 {
		return apperr.NotFound("任务不存在: " + t.ID)
	}
	return nil
}

// List 实现 TaskRepository.List。
func (r *MongoTaskRepo) List(ctx context.Context, status domain.TaskStatus) ([]*domain.Task, error) {
	filter := bson.M{}
	if status != "" {
		filter["status"] = status
	}
	cur, err := r.col.Find(ctx, filter)
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []*domain.Task
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ListByOwner 实现 TaskRepository.ListByOwner。
func (r *MongoTaskRepo) ListByOwner(ctx context.Context, owner string) ([]*domain.Task, error) {
	cur, err := r.col.Find(ctx, bson.M{"owner": owner})
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []*domain.Task
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// CountActive 实现 TaskRepository.CountActive，将过滤条件下推到 MongoDB 统计。
func (r *MongoTaskRepo) CountActive(ctx context.Context, owner, projectID string) (int, error) {
	filter := bson.M{
		"status": bson.M{"$in": bson.A{domain.TaskScheduled, domain.TaskRunning}},
	}
	if projectID != "" {
		filter["project_id"] = projectID
	} else {
		filter["owner"] = owner
	}
	n, err := r.col.CountDocuments(ctx, filter)
	if err != nil {
		return 0, err
	}
	return int(n), nil
}

// MongoUsageRepo 是 UsageRepository 的 MongoDB 实现。
type MongoUsageRepo struct{ col *mongo.Collection }

// Create 实现 UsageRepository.Create。
func (r *MongoUsageRepo) Create(ctx context.Context, u *domain.UsageRecord) error {
	_, err := r.col.InsertOne(ctx, u)
	return err
}

// List 实现 UsageRepository.List。
func (r *MongoUsageRepo) List(ctx context.Context) ([]*domain.UsageRecord, error) {
	cur, err := r.col.Find(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []*domain.UsageRecord
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ListByOwner 实现 UsageRepository.ListByOwner。
func (r *MongoUsageRepo) ListByOwner(ctx context.Context, owner string) ([]*domain.UsageRecord, error) {
	cur, err := r.col.Find(ctx, bson.M{"owner": owner})
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []*domain.UsageRecord
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// MongoQuotaRepo 是 QuotaRepository 的 MongoDB 实现。
type MongoQuotaRepo struct{ col *mongo.Collection }

// Get 实现 QuotaRepository.Get。
func (r *MongoQuotaRepo) Get(ctx context.Context, subject string) (*domain.Quota, error) {
	var q domain.Quota
	err := r.col.FindOne(ctx, bson.M{"_id": subject}).Decode(&q)
	if err == mongo.ErrNoDocuments {
		return nil, apperr.NotFound("配额不存在: " + subject)
	}
	if err != nil {
		return nil, err
	}
	return &q, nil
}

// Upsert 实现 QuotaRepository.Upsert。
func (r *MongoQuotaRepo) Upsert(ctx context.Context, q *domain.Quota) error {
	opts := options.Replace().SetUpsert(true)
	_, err := r.col.ReplaceOne(ctx, bson.M{"_id": q.Subject}, q, opts)
	return err
}

// AddUsage 实现 QuotaRepository.AddUsage。
func (r *MongoQuotaRepo) AddUsage(ctx context.Context, subject string, gpuHours float64) error {
	// 仅在配额存在时累加；不存在视为不限，忽略。
	_, err := r.col.UpdateOne(ctx, bson.M{"_id": subject},
		bson.M{"$inc": bson.M{"used_gpu_hours": gpuHours}})
	return err
}

// MongoProjectRepo 是 ProjectRepository 的 MongoDB 实现。
type MongoProjectRepo struct{ col *mongo.Collection }

// Create 实现 ProjectRepository.Create。
func (r *MongoProjectRepo) Create(ctx context.Context, p *domain.Project) error {
	_, err := r.col.InsertOne(ctx, p)
	if mongo.IsDuplicateKeyError(err) {
		return apperr.Conflict("项目已存在: " + p.ID)
	}
	return err
}

// GetByID 实现 ProjectRepository.GetByID。
func (r *MongoProjectRepo) GetByID(ctx context.Context, id string) (*domain.Project, error) {
	var p domain.Project
	err := r.col.FindOne(ctx, bson.M{"_id": id}).Decode(&p)
	if err == mongo.ErrNoDocuments {
		return nil, apperr.NotFound("项目不存在: " + id)
	}
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// Update 实现 ProjectRepository.Update。
func (r *MongoProjectRepo) Update(ctx context.Context, p *domain.Project) error {
	res, err := r.col.ReplaceOne(ctx, bson.M{"_id": p.ID}, p)
	if err != nil {
		return err
	}
	if res.MatchedCount == 0 {
		return apperr.NotFound("项目不存在: " + p.ID)
	}
	return nil
}

// ListByMember 实现 ProjectRepository.ListByMember。
func (r *MongoProjectRepo) ListByMember(ctx context.Context, userID string) ([]*domain.Project, error) {
	cur, err := r.col.Find(ctx, bson.M{"members": userID})
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []*domain.Project
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}
