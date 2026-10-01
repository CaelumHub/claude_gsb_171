# -*- coding: utf-8 -*-
"""
解释器 / 基于栈的字节码虚拟机（VM）。

执行编译器前端产出的 ``ProgramCode``（字节码）。特点：
  * 每个调用帧拥有独立的操作数栈与局部变量表，函数调用在帧间传递返回值；
  * 支持断点 / 单步（由 Debugger 驱动，run 循环在每条指令前回调 should_pause）；
  * 支持性能剖析（插桩上报 + 采样线程读取当前指令位置）；
  * 运行期错误统一转成结构化诊断（Diagnostic）抛出；
  * 具备指令数上限、调用深度上限等保护，防止死循环 / 无限递归拖垮服务。

"调试器与解释器的状态同步"的关键：VM 是**可续跑**的——每次 HTTP 请求只推进若干条
指令，帧栈、指令指针、局部变量、操作数栈、堆都完整保留在内存里，供下一请求继续。
"""

import math
import time
import random as _random
from typing import List, Dict, Any, Optional

from . import bytecode as bc
from . import runtime as rt
from . import diagnostics as diag
from .memory_model import Heap


class VMRuntimeError(Exception):
    """携带结构化诊断的运行期错误。"""

    def __init__(self, diagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class SystemExitSignal(Exception):
    """exit() 内置函数触发的正常退出信号。"""


class Frame:
    """一个调用帧。"""

    __slots__ = ("func_name", "code", "func_obj", "locals", "ip", "stack",
                 "is_main", "current_line")

    def __init__(self, func_name, code, func_obj=None, is_main=False):
        self.func_name = func_name
        self.code = code                # FunctionCode
        self.func_obj = func_obj        # RuntimeFunction 或 None（main）
        self.locals: Dict[str, Any] = {}
        self.ip = 0
        self.stack: List[Any] = []      # 本帧操作数栈
        self.is_main = is_main
        self.current_line = 0

    def __repr__(self):
        return f"Frame({self.func_name}, ip={self.ip}, locals={list(self.locals)})"


class VM:
    """MiniLang 字节码虚拟机。"""

    def __init__(self, program, source_lines=None, limits=None):
        self.program = program
        self.source_lines = source_lines or []
        self.limits = limits or {}
        self.heap = Heap()
        self.globals: Dict[str, Any] = {}
        self.builtins: Dict[str, rt.BuiltinFunction] = {}
        self.frames: List[Frame] = []
        self.instruction_count = 0
        self.instruction_limit = self.limits.get("instruction_limit", 10_000_000)
        self.max_call_depth = self.limits.get("max_call_depth", 512)
        self.finished = False
        self.paused = False
        self.pause_reason = None
        self.error: Optional[diag.Diagnostic] = None
        self.output: List[str] = []
        self.input_queue: List[str] = []
        self.return_value = None
        self.profiler = None
        self.debugger = None
        self._start_time = None
        self._cur_line = 0
        self._functions = self._bind_functions()
        self._bind_builtins()
        # 预先把用户函数对象注册进全局，实现"函数提升"
        for name, fn in self._functions.items():
            self.globals[name] = fn

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _bind_functions(self) -> Dict[str, rt.RuntimeFunction]:
        fns = {}
        for name, fc in self.program.functions.items():
            fns[name] = self.heap.allocate_function(name, fc.arity, fc.params, fc)
        return fns

    def _bind_builtins(self):
        b = {}
        b["print"] = rt.BuiltinFunction("print", self._bi_print, None)
        b["len"] = rt.BuiltinFunction("len", self._bi_len, 1)
        b["push"] = rt.BuiltinFunction("push", self._bi_push, 2)
        b["pop"] = rt.BuiltinFunction("pop", self._bi_pop, 1)
        b["type"] = rt.BuiltinFunction("type", self._bi_type, 1)
        b["str"] = rt.BuiltinFunction("str", self._bi_str, 1)
        b["int"] = rt.BuiltinFunction("int", self._bi_int, 1)
        b["float"] = rt.BuiltinFunction("float", self._bi_float, 1)
        b["range"] = rt.BuiltinFunction("range", self._bi_range, None)
        b["abs"] = rt.BuiltinFunction("abs", self._bi_abs, 1)
        b["min"] = rt.BuiltinFunction("min", self._bi_min, None, min_arity=1)
        b["max"] = rt.BuiltinFunction("max", self._bi_max, None, min_arity=1)
        b["sqrt"] = rt.BuiltinFunction("sqrt", self._bi_sqrt, 1)
        b["floor"] = rt.BuiltinFunction("floor", self._bi_floor, 1)
        b["ceil"] = rt.BuiltinFunction("ceil", self._bi_ceil, 1)
        b["round"] = rt.BuiltinFunction("round", self._bi_round, None, min_arity=1)
        b["input"] = rt.BuiltinFunction("input", self._bi_input, 0)
        b["time"] = rt.BuiltinFunction("time", self._bi_time, 0)
        b["random"] = rt.BuiltinFunction("random", self._bi_random, 0)
        b["exit"] = rt.BuiltinFunction("exit", self._bi_exit, None, min_arity=0)
        self.builtins = b

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """初始化 main 帧，准备开始执行。"""
        self.frames = []
        main = self.program.main or bc.FunctionCode("<main>", 0, [])
        if not main.is_main:
            main.is_main = True
        frame = Frame("<main>", main, is_main=True)
        self.frames.append(frame)
        self.finished = False
        self.paused = False
        self.pause_reason = None
        self.error = None
        self.instruction_count = 0
        self._start_time = time.perf_counter()
        if self.profiler:
            self.profiler.begin_run()

    def current_position(self):
        if self.frames:
            fr = self.frames[-1]
            return (fr.func_name, max(1, fr.current_line - 1))
        return ("<main>", 0)

    def peek_instruction(self):
        if self.finished or not self.frames:
            return None
        fr = self.frames[-1]
        if fr.ip < len(fr.code.instructions):
            return fr.code.instructions[fr.ip]
        return None

    def elapsed_ms(self):
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000.0

    # ------------------------------------------------------------------
    # 执行循环
    # ------------------------------------------------------------------
    def step_instruction(self):
        """执行一条指令（单步的最小粒度）。"""
        if self.finished or not self.frames:
            self.finished = True
            return
        frame = self.frames[-1]
        code = frame.code
        if frame.ip >= len(code.instructions):
            # 隐式返回
            self._do_return(frame, None)
            return
        ins = code.instructions[frame.ip]
        frame.current_line = ins.line
        self._cur_line = ins.line
        frame.ip += 1
        self.instruction_count += 1
        if self.profiler:
            self.profiler.record_instruction(frame.func_name, ins.line)
        if self.instruction_count > self.instruction_limit:
            self._runtime_error(diag.runtime_instruction_limit(
                self.instruction_limit, ins.line, 1, self._line(ins.line)))
            return
        try:
            self._dispatch(ins, frame)
        except SystemExitSignal:
            self.finished = True
        except VMRuntimeError as e:
            self._handle_runtime_error(e)
        except Exception as e:
            # 兜底：任何未被显式转成诊断的 Python 异常（内置函数域错误、
            # 类型错误等）都在此统一包装成结构化运行期诊断，绝不冒泡成 HTTP 500。
            line = getattr(ins, "line", self._cur_line) or self._cur_line or 1
            self._handle_runtime_error(VMRuntimeError(diag.runtime_internal_error(
                f"{type(e).__name__}: {e}", line, 1, self._line(line))))

    def run(self, pause_fn=None, on_pause=None, max_steps=None):
        """持续执行直到 pause_fn 返回 True、程序结束、出错、或步数耗尽。"""
        steps = 0
        while not self.finished and self.frames:
            if pause_fn is not None and pause_fn(self):
                self.paused = True
                self.pause_reason = getattr(self.debugger, "pause_reason", "pause")
                if on_pause is not None:
                    on_pause(self)
                return self
            if max_steps is not None and steps >= max_steps:
                return self
            self.step_instruction()
            steps += 1
            if self.paused:
                return self
        return self

    def _handle_runtime_error(self, e: VMRuntimeError):
        self.error = e.diagnostic
        self.finished = True
        if self.profiler:
            self.profiler.end_run()

    def _runtime_error(self, d: diag.Diagnostic):
        raise VMRuntimeError(d)

    def _line(self, line):
        if 0 <= line - 1 < len(self.source_lines):
            return self.source_lines[line - 1]
        return ""

    # ------------------------------------------------------------------
    # 指令分发
    # ------------------------------------------------------------------
    def _dispatch(self, ins: bc.Instruction, frame: Frame):
        op = ins.op
        s = frame.stack
        if op == bc.OP_LOAD_CONST:
            s.append(self._const(frame, ins.operand))
        elif op == bc.OP_LOAD_VAR:
            if ins.operand in frame.locals:
                s.append(frame.locals[ins.operand])
            elif ins.operand in self.globals:
                s.append(self.globals[ins.operand])
            else:
                self._name_error(ins, frame)
        elif op == bc.OP_STORE_VAR:
            v = s.pop()
            if ins.operand in frame.locals:
                frame.locals[ins.operand] = v
                if frame.is_main:
                    self.globals[ins.operand] = v
            else:
                self.globals[ins.operand] = v
        elif op == bc.OP_LOAD_GLOBAL:
            if ins.operand in self.globals:
                s.append(self.globals[ins.operand])
            elif ins.operand in self.builtins:
                s.append(self.builtins[ins.operand])
            else:
                self._name_error(ins, frame)
        elif op == bc.OP_STORE_GLOBAL:
            self.globals[ins.operand] = s.pop()
        elif op == bc.OP_LOAD_BUILTIN:
            s.append(self.builtins[ins.operand])
        elif op == bc.OP_LOAD_FUNC:
            s.append(self._functions[ins.operand])
        elif op == bc.OP_BINARY:
            self._binary(ins, frame)
        elif op == bc.OP_UNARY:
            self._unary(ins, frame)
        elif op == bc.OP_JUMP:
            frame.ip = ins.operand
        elif op == bc.OP_JUMP_IF_FALSE:
            v = s.pop()
            if not rt.truthy(v):
                frame.ip = ins.operand
        elif op == bc.OP_JUMP_IF_TRUE:
            v = s.pop()
            if rt.truthy(v):
                frame.ip = ins.operand
        elif op == bc.OP_POP:
            s.pop()
        elif op == bc.OP_DUP:
            s.append(s[-1])
        elif op == bc.OP_DUP2:
            a = s[-2]; b = s[-1]
            s.append(a); s.append(b)
        elif op == bc.OP_CALL:
            self._call(ins, frame)
        elif op == bc.OP_RETURN:
            self._do_return(frame, s.pop() if s else None)
        elif op == bc.OP_RETURN_NONE:
            self._do_return(frame, None)
        elif op == bc.OP_MAKE_LIST:
            items = s[-ins.operand:] if ins.operand else []
            if ins.operand:
                del s[-ins.operand:]
            s.append(self.heap.allocate_list(items))
        elif op == bc.OP_INDEX_LOAD:
            self._index_load(ins, frame)
        elif op == bc.OP_INDEX_STORE:
            self._index_store(ins, frame)
        elif op == bc.OP_NOP:
            pass
        elif op == bc.OP_LINE:
            frame.current_line = ins.operand
        else:
            self._runtime_error(diag.Diagnostic(
                diag.SEVERITY_ERROR, diag.PHASE_RUNTIME, diag.KIND_RUNTIME,
                f"未知的字节码操作码 {op!r}", ins.line, 1, 1, ins.line, 2,
                "请检查编译器产出的字节码。", None, self._line(ins.line)))

    def _const(self, frame, index):
        return frame.code.constants[index]

    # ------------------------------------------------------------------
    # 运算
    # ------------------------------------------------------------------
    def _binary(self, ins, frame):
        s = frame.stack
        right = s.pop()
        left = s.pop()
        op = ins.operand
        try:
            if op == "+":
                # 数字相加；字符串拼接
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    s.append(left + right)
                elif isinstance(left, (str, rt.RuntimeString)) or isinstance(right, (str, rt.RuntimeString)):
                    s.append(self.heap.allocate_string(_to_py_str(left) + _to_py_str(right)))
                elif isinstance(left, rt.RuntimeList) and isinstance(right, rt.RuntimeList):
                    s.append(self.heap.allocate_list(left.items + right.items))
                else:
                    self._binary_type_error("+", left, right, ins)
            elif op == "-":
                self._check_numeric(left, right, ins); s.append(left - right)
            elif op == "*":
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    s.append(left * right)
                elif isinstance(left, str) and isinstance(right, int):
                    s.append(left * right)
                else:
                    self._binary_type_error("*", left, right, ins)
            elif op == "/":
                self._check_numeric(left, right, ins)
                if right == 0:
                    self._runtime_error(diag.runtime_division_by_zero(
                        ins.line, 1, self._line(ins.line)))
                s.append(left / right)
            elif op == "%":
                self._check_numeric(left, right, ins)
                if right == 0:
                    self._runtime_error(diag.runtime_division_by_zero(
                        ins.line, 1, self._line(ins.line)))
                s.append(left % right)
            elif op == "==":
                s.append(_eq(left, right))
            elif op == "!=":
                s.append(not _eq(left, right))
            elif op == "<":
                self._check_comparable(left, right, ins); s.append(left < right)
            elif op == "<=":
                self._check_comparable(left, right, ins); s.append(left <= right)
            elif op == ">":
                self._check_comparable(left, right, ins); s.append(left > right)
            elif op == ">=":
                self._check_comparable(left, right, ins); s.append(left >= right)
            else:
                self._runtime_error(diag.Diagnostic(
                    diag.SEVERITY_ERROR, diag.PHASE_RUNTIME, diag.KIND_RUNTIME,
                    f"未知运算符 {op!r}", ins.line, 1, 1, ins.line, 2,
                    "", None, self._line(ins.line)))
        except VMRuntimeError:
            raise
        except Exception:
            self._runtime_error(diag.runtime_binary_type_error(
                op, rt.type_name(left), rt.type_name(right),
                ins.line, 1, self._line(ins.line)))

    def _unary(self, ins, frame):
        s = frame.stack
        v = s.pop()
        if ins.operand == "-":
            if isinstance(v, (int, float)):
                s.append(-v)
            else:
                self._type_error(ins, "数字", rt.type_name(v))
        elif ins.operand == "!":
            s.append(not rt.truthy(v))

    def _index_load(self, ins, frame):
        s = frame.stack
        idx = s.pop()
        target = s.pop()
        self._check_indexable(target, ins)
        try:
            if isinstance(target, rt.RuntimeList):
                s.append(target.items[idx])
            elif isinstance(target, (str, rt.RuntimeString)):
                s.append(self.heap.allocate_string(_to_py_str(target)[idx]))
        except (IndexError, KeyError):
            self._runtime_error(diag.runtime_index_out_of_range(
                idx, len(target.items) if isinstance(target, rt.RuntimeList) else len(target),
                ins.line, 1, self._line(ins.line)))
        except TypeError:
            self._type_error(ins, "整数下标", rt.type_name(idx))

    def _index_store(self, ins, frame):
        s = frame.stack
        value = s.pop()
        idx = s.pop()
        target = s.pop()
        if not isinstance(target, rt.RuntimeList):
            self._type_error(ins, "列表", rt.type_name(target))
        if not isinstance(idx, int) or isinstance(idx, bool):
            self._type_error(ins, "整数下标", rt.type_name(idx))
        if not (0 <= idx < len(target.items)):
            self._runtime_error(diag.runtime_index_out_of_range(
                idx, len(target.items), ins.line, 1, self._line(ins.line)))
        target.items[idx] = value

    # ------------------------------------------------------------------
    # 调用与返回
    # ------------------------------------------------------------------
    def _call(self, ins, frame):
        argc = ins.operand
        args = [frame.stack.pop() for _ in range(argc)]
        args.reverse()
        callee = frame.stack.pop()
        if isinstance(callee, rt.BuiltinFunction):
            self._check_builtin_arity(callee, argc, ins)
            if self.profiler:
                self.profiler.function_enter("builtin:" + callee.name)
            try:
                result = callee.fn(args)
            except VMRuntimeError:
                raise
            except (TypeError, ValueError, OverflowError, ArithmeticError) as e:
                # 内置函数底层（math/round/min/max 等）抛出的 Python 异常，
                # 统一转成结构化的"非法参数"诊断，避免冒泡成服务器 500。
                self._builtin_invalid_arg(callee.name, args, e, ins)
            finally:
                if self.profiler:
                    self.profiler.function_exit()
            frame.stack.append(result)
            return
        if isinstance(callee, rt.RuntimeFunction):
            self._call_user(ins, frame, callee, args)
            return
        self._type_error(ins, "函数", rt.type_name(callee))

    def _check_builtin_arity(self, callee, argc, ins):
        """运行期参数个数校验（间接调用无法在语义阶段拦截，这里兜底）。"""
        if callee.arity is not None and argc != callee.arity:
            self._runtime_error(diag.runtime_wrong_arity(
                callee.name, callee.arity, argc, ins.line, 1, self._line(ins.line)))
        if callee.min_arity is not None and argc < callee.min_arity:
            self._runtime_error(diag.runtime_wrong_arity(
                callee.name, None, argc, ins.line, 1, self._line(ins.line),
                accepted=f"至少 {callee.min_arity} "))

    def _builtin_invalid_arg(self, name, args, exc, ins):
        """把内置函数收到的非法实参映射为统一风格的运行期诊断。"""
        line = ins.line
        src = self._line(line)
        # 优先按"取值越界"（domain）与"类型不对"（type）给出针对性文案
        detail = self._builtin_arg_detail(name, args, exc)
        if detail is not None:
            message, fix = detail
            self._runtime_error(diag.runtime_builtin_domain_error(
                name, message, fix, line, 1, src))
        expected = self._builtin_expected_types(name, args)
        actual = "、".join(rt.type_name(a) for a in args) if args else "（无参数）"
        self._runtime_error(diag.runtime_wrong_type(
            expected, actual, line, 1, src, op=f"函数 {name}()"))

    def _builtin_expected_types(self, name, args):
        """兜底类型诊断时，给出该内置函数期望的参数类型描述。"""
        numeric_fns = ("sqrt", "floor", "ceil", "abs")
        if name in numeric_fns:
            return "数值（int 或 float）"
        if name == "round":
            return "数值（可选第二个整数位数参数）"
        if name in ("min", "max"):
            return "类型一致、可相互比较的值"
        if name == "range":
            return "整数（起点、终点、步长）"
        return "类型与函数签名匹配的值"

    def _builtin_arg_detail(self, name, args, exc):
        """为可判定的越界 / 非法取值场景返回 (message, fix)，否则返回 None。"""
        x = args[0] if args else None
        numeric = isinstance(x, (int, float)) and not isinstance(x, bool)
        if name == "sqrt" and numeric:
            return ("平方根的被开方数不能是负数",
                   "sqrt() 只接受非负数（x >= 0）；若需要处理负数，请先取绝对值 abs(x)。")
        if name in ("floor", "ceil") and numeric:
            return ("参数数值超出可表示范围",
                    "检查传入的数值是否过大，先把它限制在合理范围内。")
        if name == "round" and numeric:
            if len(args) == 2 and not isinstance(args[1], int):
                return ("round() 的第二个参数（保留位数）必须是整数",
                        "把保留位数改成整数字面量，例如 round(x, 2)。")
            return ("参数数值超出可表示范围",
                    "检查传入的数值是否过大或不是有限数。")
        if name in ("min", "max"):
            return (f"{name}() 只能在相互可比较的值之间取{'最小' if name == 'min' else '最大'}值",
                    "确保传入的值类型一致（全是数字或全是字符串），不要混用数字与字符串。")
        if name == "range":
            return ("range() 的起点、终点、步长都必须是整数，且步长不能为 0",
                    "把实参改为整数；若用了变量，先用 int() 转换，并确保步长不为 0。")
        if name == "abs" and numeric:
            return ("参数数值超出可表示范围",
                    "检查传入的数值是否过大。")
        if isinstance(exc, (ValueError, OverflowError)) and numeric:
            return ("参数取值超出该函数允许的范围",
                    "检查参数的取值范围，先用条件判断保护，再调用该函数。")
        return None

    def _call_user(self, ins, frame, func, args):
        if len(args) != func.arity:
            self._runtime_error(diag.runtime_wrong_arity(
                func.name, func.arity, len(args), ins.line, 1, self._line(ins.line)))
        if len(self.frames) >= self.max_call_depth:
            self._runtime_error(diag.runtime_stack_overflow(
                self.max_call_depth, ins.line, 1, self._line(ins.line)))
        new_frame = Frame(func.name, func.code, func_obj=func)
        for i, p in enumerate(func.params):
            new_frame.locals[p] = args[i]
        if self.profiler:
            self.profiler.function_enter(func.name)
        self.frames.append(new_frame)

    def _do_return(self, frame, value):
        # 退出当前帧，把返回值交给上一帧
        if self.profiler and not frame.is_main:
            self.profiler.function_exit()
        if len(self.frames) == 1:
            # main 返回 -> 程序结束
            self.return_value = value
            self.frames.pop()
            self.finished = True
            if self.profiler:
                self.profiler.end_run()
            return
        # 弹出当前帧
        caller_index = len(self.frames) - 2
        self.frames.pop()
        if self.frames:
            self.frames[caller_index].stack.append(value)

    # ------------------------------------------------------------------
    # 名称 / 类型错误
    # ------------------------------------------------------------------
    def _name_error(self, ins, frame):
        name = ins.operand
        candidates = set(self.globals.keys()) | set(self.builtins.keys())
        self._runtime_error(diag.runtime_name_error(
            name, sorted(candidates), ins.line, 1, self._line(ins.line)))

    def _type_error(self, ins, expected, actual):
        self._runtime_error(diag.runtime_wrong_type(
            expected, actual, ins.line, 1, self._line(ins.line)))

    def _binary_type_error(self, op, left, right, ins):
        self._runtime_error(diag.runtime_binary_type_error(
            op, rt.type_name(left), rt.type_name(right),
            ins.line, 1, self._line(ins.line)))

    def _check_numeric(self, a, b, ins):
        if not (isinstance(a, (int, float)) and not isinstance(a, bool)
                and isinstance(b, (int, float)) and not isinstance(b, bool)):
            self._binary_type_error(ins.operand, a, b, ins)

    def _check_comparable(self, a, b, ins):
        if not ((isinstance(a, (int, float, str, rt.RuntimeString)) and
                 isinstance(b, (int, float, str, rt.RuntimeString))) or
                (isinstance(a, (int, float)) and isinstance(b, (int, float)))):
            self._binary_type_error(ins.operand, a, b, ins)

    def _check_indexable(self, target, ins):
        if not isinstance(target, (rt.RuntimeList, str, rt.RuntimeString)):
            self._type_error(ins, "列表或字符串", rt.type_name(target))

    # ------------------------------------------------------------------
    # 状态快照（供调试器 / 内存模型）
    # ------------------------------------------------------------------
    def frame_snapshot(self):
        frames = []
        for fr in reversed(self.frames):
            locals_ = {k: rt.serialize_value(v) for k, v in fr.locals.items()}
            frames.append({
                "function": fr.func_name,
                "line": fr.current_line,
                "ip": max(0, fr.ip - 1),
                "is_main": fr.is_main,
                "locals": locals_,
            })
        return frames

    def stack_snapshot(self):
        """返回当前帧操作数栈的内容（用于调试观察）。"""
        if not self.frames:
            return []
        fr = self.frames[-1]
        return [rt.serialize_value(v) for v in fr.stack]

    # ------------------------------------------------------------------
    # 内置函数
    # ------------------------------------------------------------------
    def _bi_print(self, args):
        parts = []
        for a in args:
            parts.append(_to_display(a))
        line = " ".join(parts)
        self.output.append(line)
        return None

    def _bi_len(self, args):
        v = args[0]
        if isinstance(v, rt.RuntimeList):
            return len(v.items)
        if isinstance(v, (str, rt.RuntimeString)):
            return len(_to_py_str(v))
        self._builtin_type_error("列表或字符串", rt.type_name(v), op="len()")

    def _bi_push(self, args):
        lst, val = args
        if not isinstance(lst, rt.RuntimeList):
            self._builtin_type_error("列表", rt.type_name(lst), op="push() 的第一个参数")
        lst.items.append(val)
        lst.size = len(lst.items)
        return lst

    def _bi_pop(self, args):
        lst = args[0]
        if not isinstance(lst, rt.RuntimeList):
            self._builtin_type_error("列表", rt.type_name(lst), op="pop()")
        if not lst.items:
            return None
        return lst.items.pop()

    def _bi_type(self, args):
        return rt.type_name(args[0])

    def _bi_str(self, args):
        return _to_py_str(args[0])

    def _bi_int(self, args):
        v = args[0]
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, (str, rt.RuntimeString)):
            try:
                return int(_to_py_str(v).strip())
            except ValueError:
                return 0
        return 0

    def _bi_float(self, args):
        v = args[0]
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, (str, rt.RuntimeString)):
            try:
                return float(_to_py_str(v).strip())
            except ValueError:
                return 0.0
        return 0.0

    def _bi_range(self, args):
        start, stop, step = 0, 0, 1
        if len(args) == 1:
            stop = args[0]
        elif len(args) == 2:
            start, stop = args
        elif len(args) == 3:
            start, stop, step = args
        else:
            self._runtime_error(diag.runtime_wrong_arity(
                "range", None, len(args), self._cur_line or 1, 1,
                self._line(self._cur_line or 1), accepted="1~3 "))
        return self.heap.allocate_list(list(range(start, stop + 1, step)))

    def _bi_abs(self, args):
        return abs(args[0])

    def _bi_min(self, args):
        return min(args)

    def _bi_max(self, args):
        return max(args)

    def _bi_sqrt(self, args):
        return math.sqrt(args[0])

    def _bi_floor(self, args):
        return int(math.floor(args[0]))

    def _bi_ceil(self, args):
        return int(math.ceil(args[0]))

    def _bi_round(self, args):
        if not args:
            self._runtime_error(diag.runtime_wrong_arity(
                "round", 1, 0, self._cur_line or 1, 1,
                self._line(self._cur_line or 1)))
        if len(args) == 2:
            return round(args[0], args[1])
        return round(args[0], 1)

    def _bi_input(self, args):
        if self.input_queue:
            return self.input_queue.pop(0)
        return ""

    def _bi_time(self, args):
        return time.time()

    def _bi_random(self, args):
        return _random.random()

    def _bi_exit(self, args):
        code = args[0] if args else 0
        self.return_value = code
        raise SystemExitSignal()

    def _builtin_type_error(self, expected, actual, op=None):
        line = self._cur_line or 1
        self._runtime_error(diag.runtime_wrong_type(
            expected, actual, line, 1, self._line(line), op=op or "该函数"))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _to_py_str(v) -> str:
    if isinstance(v, rt.RuntimeString):
        return v.value
    if isinstance(v, str):
        return v
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _to_display(v) -> str:
    if isinstance(v, rt.RuntimeList):
        return "[" + ", ".join(_to_display(x) for x in v.items) + "]"
    if isinstance(v, rt.RuntimeString):
        return v.value
    if isinstance(v, str):
        return v
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == int(v):
            return f"{v:.1f}"
        return repr(v)
    return str(v)


def _eq(a, b) -> bool:
    if isinstance(a, rt.RuntimeList) and isinstance(b, rt.RuntimeList):
        return a is b  # 引用相等（MiniLang 列表比较为身份比较）
    if isinstance(a, rt.RuntimeString):
        a = a.value
    if isinstance(b, rt.RuntimeString):
        b = b.value
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b
