package proxy

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Every outcome that counts against a credential must go through recordBanSignal
// (which steps aside when the health worker owns the decisions) and reportHealth
// (which feeds the worker). A direct balancer.RecordResponse anywhere else would
// keep counting locally even with account_events on — and the worker's ban reader,
// seeing no such ban in its snapshot, would lift it on the next sync, so the ban
// would silently not hold. This guards every error path added from here on: the
// only place allowed to call RecordResponse is recordBanSignal itself.
func TestNoUngatedBalancerRecordResponseInProxyPackage(t *testing.T) {
	fset := token.NewFileSet()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}

	var offenders []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, filepath.Join(".", name), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv == nil || fn.Name.Name == "recordBanSignal" {
				continue
			}
			ast.Inspect(fn, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := call.Fun.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				if sel.Sel.Name != "RecordResponse" {
					return true
				}
				// Match `p.balancer.RecordResponse` / `c.p.balancer.RecordResponse`.
				if inner, ok := sel.X.(*ast.SelectorExpr); ok && inner.Sel.Name == "balancer" {
					pos := fset.Position(call.Pos())
					offenders = append(offenders, pos.String()+" in "+fn.Name.Name)
				}
				return true
			})
		}
	}
	if len(offenders) > 0 {
		t.Fatalf("balancer.RecordResponse called outside recordBanSignal (route it through recordBanSignal + reportHealth):\n  %s",
			strings.Join(offenders, "\n  "))
	}
}
