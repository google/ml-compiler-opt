#ifndef COMPILER_OPT_MEMTRACE_COSTMODEL_STATUS_MACROS_H_
#define COMPILER_OPT_MEMTRACE_COSTMODEL_STATUS_MACROS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define RETURN_IF_ERROR(expr) \
  if (auto _status = (expr); !_status.ok()) return _status;

#define ASSIGN_OR_RETURN_IMPL(tmp, lhs, rexpr) \
  auto tmp = (rexpr);                          \
  if (!tmp.ok()) return tmp.status();          \
  lhs = std::move(*tmp);

#define ASSIGN_OR_RETURN(lhs, rexpr) \
  ASSIGN_OR_RETURN_IMPL(CONCAT(_status_or_, __LINE__), lhs, rexpr)

#endif  // COMPILER_OPT_MEMTRACE_COSTMODEL_STATUS_MACROS_H_
