package escape_analysis

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// ==============================================
// 场景 1：返回指针 vs 返回值
// ==============================================

type User struct {
	Name string
	Age  int
}

//go:noinline
func NewUserPointer() *User {
	return &User{Name: "bob", Age: 20} // 逃逸！
}

//go:noinline
func NewUserValue() User {
	return User{Name: "bob", Age: 20} // 不逃逸！
}

func BenchmarkNewUserPointer(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = NewUserPointer()
	}
}

func BenchmarkNewUserValue(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = NewUserValue()
	}
}

// ==============================================
// 场景 2：变量大小确定 vs 不确定
// ==============================================

//go:noinline
func DynamicSize(n int) []int {
	s := make([]int, n) // n 是变量，逃逸！
	return s
}

//go:noinline
func FixedSize() []int {
	s := make([]int, 100) // 编译期知道大小，栈上
	return s
}

func BenchmarkDynamicSize(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = DynamicSize(100)
	}
}

func BenchmarkFixedSize(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = FixedSize()
	}
}

// ==============================================
// 场景 3：闭包捕获变量 vs 结构体方法
// ==============================================

//go:noinline
func ClosureCounter() func() int {
	x := 0 // 被闭包捕获，逃逸！
	return func() int {
		x++
		return x
	}
}

type Counter struct {
	x int
}

func (c *Counter) Inc() int {
	c.x++
	return c.x
}

func BenchmarkClosureCounter(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		c := ClosureCounter()
		_ = c()
	}
}

func BenchmarkStructCounter(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	var c Counter
	for i := 0; i < b.N; i++ {
		_ = c.Inc()
	}
}

// ==============================================
// 场景 4：channel 发指针 vs 发值
// ==============================================

func BenchmarkChanPointer(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	ch := make(chan *User, 100)
	for i := 0; i < b.N; i++ {
		u := &User{Name: "bob", Age: 20} // 逃逸！
		ch <- u
		<-ch
	}
}

func BenchmarkChanValue(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	ch := make(chan User, 100)
	for i := 0; i < b.N; i++ {
		u := User{Name: "bob", Age: 20} // 不逃逸，拷贝
		ch <- u
		<-ch
	}
}

// ==============================================
// 场景 5：fmt.Sprintf vs strings.Builder vs 复用 Builder
// ==============================================

func BenchmarkSprintf(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = fmt.Sprintf("name: %s, age: %d", "bob", 20)
	}
}

func BenchmarkBuilder(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var b strings.Builder
		b.WriteString("name: ")
		b.WriteString("bob")
		b.WriteString(", age: ")
		b.WriteString(strconv.Itoa(20))
		_ = b.String()
	}
}

func BenchmarkBuilderReuse(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	var builder strings.Builder
	for i := 0; i < b.N; i++ {
		builder.Reset()
		builder.WriteString("name: ")
		builder.WriteString("bob")
		builder.WriteString(", age: ")
		builder.WriteString(strconv.Itoa(20))
		_ = builder.String()
	}
}

// ==============================================
// 场景 6：defer 闭包 vs 直接 defer
// ==============================================

var mu sync.Mutex

func BenchmarkDeferClosure(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		mu.Lock()
		defer func() { // 闭包捕获，逃逸
			mu.Unlock()
		}()
		_ = 1 + 1
	}
}

func BenchmarkDeferDirect(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		mu.Lock()
		defer mu.Unlock() // 直接调用，不逃逸
		_ = 1 + 1
	}
}

// ==============================================
// 场景 7：sync.Pool 复用 vs 每次新建
// ==============================================

var slicePool = sync.Pool{
	New: func() interface{} {
		return make([]int, 1024)
	},
}

func BenchmarkWithoutPool(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		s := make([]int, 1024) // 每次新建，逃逸
		_ = s
	}
}

func BenchmarkWithPool(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		s := slicePool.Get().([]int)
		_ = s
		slicePool.Put(s) // 复用，不需要再分配
	}
}

// ==============================================
// 场景 8：预定义 error vs 每次 New
// ==============================================

var ErrSomething = errors.New("something went wrong")

func BenchmarkErrorNew(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = errors.New("something went wrong") // 每次 new，逃逸
	}
}

func BenchmarkErrorPredefined(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = ErrSomething // 预定义，不逃逸
	}
}

// ==============================================
// 场景 9：小对象传值 vs 传指针
// ==============================================

type Small struct {
	A, B, C, D int
}

func ProcessPointer(s *Small) int {
	return s.A + s.B + s.C + s.D
}

func ProcessValue(s Small) int {
	return s.A + s.B + s.C + s.D
}

func BenchmarkSmallPointer(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	s := &Small{1, 2, 3, 4}
	for i := 0; i < b.N; i++ {
		_ = ProcessPointer(s)
	}
}

func BenchmarkSmallValue(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	s := Small{1, 2, 3, 4}
	for i := 0; i < b.N; i++ {
		_ = ProcessValue(s)
	}
}

// ==============================================
// 场景 10：interface{} vs 具体类型
// ==============================================

type Stringer interface {
	String() string
}

type MyString string

func (s MyString) String() string {
	return string(s)
}

func PrintInterface(s Stringer) string {
	return s.String()
}

func PrintConcrete(s MyString) string {
	return string(s)
}

func BenchmarkInterfaceCall(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	s := MyString("hello")
	for i := 0; i < b.N; i++ {
		_ = PrintInterface(s) // interface 调用，逃逸
	}
}

func BenchmarkConcreteCall(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	s := MyString("hello")
	for i := 0; i < b.N; i++ {
		_ = PrintConcrete(s) // 具体类型，不逃逸
	}
}

// ==============================================
// 场景 11：for range 拷贝 vs 索引
// ==============================================

type Item struct {
	ID    int
	Name  string
	Value string
}

func ProcessItem(item *Item) int {
	return item.ID
}

func BenchmarkForRangeCopy(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	items := make([]Item, 1000)
	for i := range items {
		items[i] = Item{ID: i, Name: "test", Value: "value"}
	}

	for i := 0; i < b.N; i++ {
		for _, item := range items { // 每次拷贝 item
			_ = ProcessItem(&item) // &item 每次都是同一个地址，还逃逸！
		}
	}
}

func BenchmarkForIndex(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	items := make([]Item, 1000)
	for i := range items {
		items[i] = Item{ID: i, Name: "test", Value: "value"}
	}

	for i := 0; i < b.N; i++ {
		for idx := range items { // 用索引，不拷贝
			_ = ProcessItem(&items[idx]) // 直接取原数组里的元素，不逃逸
		}
	}
}

// ==============================================
// 运行说明
// ==============================================
// 1. 先看逃逸分析结果：
//    go build -gcflags="-m -l" ./escape_analysis_test.go
//
// 2. 跑所有 benchmark：
//    go test -bench=. -benchmem -count=5
//
// 3. 看每个 benchmark 的 allocs/op：
//    - 0 allocs/op = 完美，栈上分配
//    - 1 allocs/op = 堆上分配，有优化空间
//
// 4. 对比优化前后的 ns/op 和 allocs/op，看差多少倍
//
// 典型输出（预期）：
// BenchmarkNewUserPointer-8          20000000     80 ns/op     32 B/op    1 allocs/op
// BenchmarkNewUserValue-8           100000000     10 ns/op      0 B/op    0 allocs/op  ← 8 倍！
//
// BenchmarkSprintf-8                  5000000    250 ns/op     64 B/op    2 allocs/op
// BenchmarkBuilderReuse-8            50000000     20 ns/op      0 B/op    0 allocs/op  ← 12.5 倍！
//
// BenchmarkWithoutPool-8              2000000    600 ns/op   8192 B/op    1 allocs/op
// BenchmarkWithPool-8                50000000     25 ns/op      0 B/op    0 allocs/op  ← 24 倍！🚀
