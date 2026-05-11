.PHONY: build test lint vet fmt all check-go check-lint

all: check-go check-lint fmt vet lint test build

check-go:
	@v=$$(go version | awk '{print $$3}' | sed 's/^go//'); \
	  case "$$v" in \
	    1.23*|1.24*|1.25*|1.26*|1.27*|1.28*|1.29*) ;; \
	    *) echo "go 1.23+ required, found $$v"; exit 1 ;; \
	  esac

check-lint:
	@command -v golangci-lint >/dev/null 2>&1 || { \
	  echo "golangci-lint required; install: brew install golangci-lint"; exit 1; }

build:
	go build ./...

test:
	go test -race -count=1 ./...

vet:
	go vet ./...

fmt:
	gofmt -l -w .

lint:
	golangci-lint run ./...
