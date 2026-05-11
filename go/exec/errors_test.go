package exec

import (
	"errors"
	"testing"
)

func TestErrCategoriesDistinct(t *testing.T) {
	cats := []error{
		ErrLLMTransient, ErrLLMFatal, ErrToolDenied, ErrToolFailed,
		ErrIRCSend, ErrCacheStale, ErrBudgetExceeded,
	}
	for i, a := range cats {
		if a == nil {
			t.Errorf("cat %d nil", i)
		}
		for j, b := range cats {
			if i == j {
				continue
			}
			if errors.Is(a, b) {
				t.Errorf("%v incorrectly Is(%v)", a, b)
			}
		}
	}
}
