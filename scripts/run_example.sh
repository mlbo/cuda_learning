#!/bin/bash
# ============================================================================
# CUDA 示例运行脚本
# 功能：快速运行对应章节的示例代码
# 用法：./run_example.sh <章节号> [参数...]
# ============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(dirname "$SCRIPT_DIR")/examples/build"

# 章节配置 (章节号 => 目录名:章节描述)
declare -A CHAPTERS=(
    # 基础篇 (1-11)
    ["1"]="01_gpu_basics:GPU 并行计算基础"
    ["2"]="02_cuda_intro:CUDA 基础概念"
    ["3"]="03_gpu_hardware:GPU 硬件架构"
    ["4"]="04_thread_hierarchy:线程层级结构"
    ["5"]="05_hello_cuda:第一个 CUDA 程序"
    ["6"]="06_memory_basics:内存管理基础"
    ["7"]="07_kernel_deep:核函数深入"
    ["8"]="08_profiling:性能分析入门"
    ["9"]="09_memory_opt:内存访问优化"
    ["10"]="10_precision:精度与性能"
    ["11"]="11_roofline:Roofline 模型"

    # 进阶篇 (12-19)
    ["12"]="12_atomic_race:原子操作与竞争条件"
    ["13"]="13_shared_memory:共享内存深入"
    ["14"]="14_reduce:规约算法优化"
    ["15"]="15_bank_conflict:Bank Conflict 优化"
    ["16"]="16_cooperative_groups:Cooperative Groups"
    ["17"]="17_gemm_basics:GEMM 优化入门"
    ["18"]="18_gemm_tiling:GEMM 分块优化"
    ["19"]="19_tensor_core:Tensor Core 编程"

    # 高级篇 (20-25)
    ["20"]="20_convolution:卷积算子实现"
    ["21"]="21_async_execution:异步执行与延迟隐藏"
    ["22"]="22_streams:CUDA 流与并发"
    ["23"]="23_data_transfer:数据传输优化"
    ["24"]="24_cuda_graph:CUDA Graph"
    ["25"]="25_multi_gpu:多 GPU 编程"

    # 专家篇 (26-30)
    ["26"]="26_quantization:低精度与量化"
    ["27"]="27_ptx_optimization:PTX 与底层优化"
    ["28"]="28_micro_optimization:微指令级调优"
    ["29"]="29_ilp_divergence:ILP 与 Warp Divergence"
    ["30"]="30_cuda_libraries:CUDA 官方库实战"
)

# ============================================================================
# 获取章节的所有可执行文件
# ============================================================================
get_chapter_executables() {
    local chapter=$1
    local dir="${CHAPTERS[$chapter]%%:*}"
    local chapter_dir="$BUILD_DIR/$dir"

    if [ -d "$chapter_dir" ]; then
        find "$chapter_dir" -maxdepth 1 -type f -executable -name "${chapter}_*" | sort
    fi
}

# ============================================================================
# 获取示例描述（从文件名提取）
# ============================================================================
get_example_desc() {
    local exe_name=$(basename "$1")
    # 提取序号后的描述部分
    local desc="${exe_name#[0-9]*_}"
    echo "${desc//_/ }"
}

# ============================================================================
# 显示帮助
# ============================================================================
show_help() {
    echo -e "${CYAN}CUDA 示例运行脚本${NC}"
    echo ""
    echo "用法: $0 <章节号> [示例序号] [参数...]"
    echo "      $0 --all-chapters        # 运行所有章节的所有示例"
    echo "      $0 --chapter-all <章节>  # 运行指定章节的所有示例"
    echo ""
    echo -e "${GREEN}基础篇 (1-11):${NC}"
    echo "-------------------------------------------"
    for chapter in 1 2 3 4 5 6 7 8 9 10 11; do
        if [ -n "${CHAPTERS[$chapter]}" ]; then
            IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"
            printf "  ${GREEN}%2s${NC}  %s\n" "$chapter" "$desc"
        fi
    done

    echo ""
    echo -e "${GREEN}进阶篇 (12-19):${NC}"
    echo "-------------------------------------------"
    for chapter in 12 13 14 15 16 17 18 19; do
        if [ -n "${CHAPTERS[$chapter]}" ]; then
            IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"
            printf "  ${GREEN}%2s${NC}  %s\n" "$chapter" "$desc"
        fi
    done

    echo ""
    echo -e "${GREEN}高级篇 (20-25):${NC}"
    echo "-------------------------------------------"
    for chapter in 20 21 22 23 24 25; do
        if [ -n "${CHAPTERS[$chapter]}" ]; then
            IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"
            printf "  ${GREEN}%2s${NC}  %s\n" "$chapter" "$desc"
        fi
    done

    echo ""
    echo -e "${GREEN}专家篇 (26-30):${NC}"
    echo "-------------------------------------------"
    for chapter in 26 27 28 29 30; do
        if [ -n "${CHAPTERS[$chapter]}" ]; then
            IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"
            printf "  ${GREEN}%2s${NC}  %s\n" "$chapter" "$desc"
        fi
    done

    echo ""
    echo "选项:"
    echo "  --list, -l              列出所有可执行文件"
    echo "  --list-chapter <章节>   列出指定章节的所有示例"
    echo "  --all, -a               运行所有章节（每个章节第一个示例）"
    echo "  --all-chapters          运行所有章节的所有示例"
    echo "  --chapter-all <章节>    运行指定章节的所有示例"
    echo "  --all-basic             运行基础篇 (1-11) 所有示例"
    echo "  --all-intermediate      运行进阶篇 (12-19) 所有示例"
    echo "  --all-advanced          运行高级篇 (20-25) 所有示例"
    echo "  --all-expert            运行专家篇 (26-30) 所有示例"
    echo "  --help, -h              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 5              # 运行第 5 章第一个示例"
    echo "  $0 12 2           # 运行第 12 章第 2 个示例 (02_atomic_add)"
    echo "  $0 --chapter-all 12   # 运行第 12 章所有示例"
    echo "  $0 --all-chapters     # 运行所有章节的所有示例"
}

# ============================================================================
# 列出指定章节的所有示例
# ============================================================================
list_chapter() {
    local chapter=$1
    if [ -z "${CHAPTERS[$chapter]}" ]; then
        echo -e "${RED}错误: 无效的章节号 '$chapter'${NC}"
        return 1
    fi

    IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"
    echo -e "${CYAN}第 $chapter 章: $desc${NC}"
    echo "目录: $dir"
    echo ""

    local exes=$(get_chapter_executables "$chapter")
    if [ -z "$exes" ]; then
        echo -e "${YELLOW}该章节没有已编译的示例${NC}"
        return 1
    fi

    local idx=1
    while IFS= read -r exe; do
        local exe_name=$(basename "$exe")
        local size=$(ls -lh "$exe" | awk '{print $5}')
        printf "  ${GREEN}%2d${NC}  %-40s %s\n" "$idx" "$exe_name" "[$size]"
        ((idx++))
    done <<< "$exes"
}

# ============================================================================
# 列出所有可执行文件
# ============================================================================
list_executables() {
    echo -e "${CYAN}已编译的可执行文件:${NC}"
    echo ""

    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}构建目录不存在，请先编译项目${NC}"
        echo -e "${YELLOW}  cd examples && mkdir -p build && cd build && cmake .. && make -j\$(nproc)${NC}"
        return 1
    fi

    local total=0

    for chapter in $(echo "${!CHAPTERS[@]}" | tr ' ' '\n' | sort -n); do
        IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"
        local exes=$(get_chapter_executables "$chapter")

        if [ -n "$exes" ]; then
            echo -e "${GREEN}第 $chapter 章: $desc${NC}"
            while IFS= read -r exe; do
                local exe_name=$(basename "$exe")
                local size=$(ls -lh "$exe" | awk '{print $5}')
                printf "    %-40s %s\n" "$exe_name" "[$size]"
                ((total++))
            done <<< "$exes"
            echo ""
        fi
    done

    echo -e "总计: ${GREEN}$total${NC} 个示例"
}

# ============================================================================
# 运行单个示例
# ============================================================================
run_single_example() {
    local exe_path=$1
    shift

    local exe_name=$(basename "$exe_path")
    local desc=$(get_example_desc "$exe_path")

    echo -e "${BLUE}执行: $exe_path $*${NC}"
    echo ""

    "$exe_path" "$@"

    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[✓] 示例运行成功${NC}"
    else
        echo -e "${RED}[✗] 示例运行失败 (退出码: $exit_code)${NC}"
    fi

    return $exit_code
}

# ============================================================================
# 运行指定章节的单个示例
# ============================================================================
run_example() {
    local chapter=$1
    local example_idx=${2:-1}
    shift 2 2>/dev/null || shift $#

    # 检查章节号是否有效
    if [ -z "${CHAPTERS[$chapter]}" ]; then
        echo -e "${RED}错误: 无效的章节号 '$chapter'${NC}"
        echo ""
        show_help
        exit 1
    fi

    IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"

    # 获取该章节所有示例
    local exes=$(get_chapter_executables "$chapter")
    if [ -z "$exes" ]; then
        echo -e "${RED}错误: 该章节没有已编译的示例${NC}"
        echo "目录: $BUILD_DIR/$dir"
        exit 1
    fi

    # 转换为数组并获取指定索引的示例
    local exe_array=()
    while IFS= read -r exe; do
        exe_array+=("$exe")
    done <<< "$exes"

    if [ "$example_idx" -lt 1 ] || [ "$example_idx" -gt ${#exe_array[@]} ]; then
        echo -e "${RED}错误: 示例序号 '$example_idx' 无效 (有效范围: 1-${#exe_array[@]})${NC}"
        echo ""
        list_chapter "$chapter"
        exit 1
    fi

    local exe_path="${exe_array[$((example_idx-1))]}"
    local exe_name=$(basename "$exe_path")

    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  第 $chapter 章: $desc - $exe_name${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo ""

    run_single_example "$exe_path" "$@"
}

# ============================================================================
# 运行指定章节的所有示例
# ============================================================================
run_chapter_all() {
    local chapter=$1

    if [ -z "${CHAPTERS[$chapter]}" ]; then
        echo -e "${RED}错误: 无效的章节号 '$chapter'${NC}"
        exit 1
    fi

    IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"
    local exes=$(get_chapter_executables "$chapter")

    if [ -z "$exes" ]; then
        echo -e "${YELLOW}第 $chapter 章没有已编译的示例${NC}"
        return 0
    fi

    local success=0
    local failed=0

    while IFS= read -r exe_path; do
        local exe_name=$(basename "$exe_path")
        echo -e "${CYAN}============================================${NC}"
        echo -e "${CYAN}  第 $chapter 章: $desc - $exe_name${NC}"
        echo -e "${CYAN}============================================${NC}"
        echo ""

        if run_single_example "$exe_path"; then
            ((success++))
        else
            ((failed++))
        fi

        echo ""
        echo "-------------------------------------------"
        echo ""
    done <<< "$exes"

    echo -e "${CYAN}第 $chapter 章结果汇总:${NC}"
    echo -e "  成功: ${GREEN}$success${NC}"
    echo -e "  失败: ${RED}$failed${NC}"

    return $failed
}

# ============================================================================
# 运行指定范围的所有示例
# ============================================================================
run_range() {
    local start=$1
    local end=$2
    local title=$3

    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  $title${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo ""

    local total_success=0
    local total_failed=0
    local total_skipped=0

    for chapter in $(seq $start $end); do
        if [ -n "${CHAPTERS[$chapter]}" ]; then
            IFS=':' read -r dir desc <<< "${CHAPTERS[$chapter]}"
            local exes=$(get_chapter_executables "$chapter")

            if [ -z "$exes" ]; then
                echo -e "${YELLOW}第 $chapter 章跳过 (无已编译示例)${NC}"
                ((total_skipped++))
                continue
            fi

            local chapter_success=0
            local chapter_failed=0

            while IFS= read -r exe_path; do
                local exe_name=$(basename "$exe_path")
                echo -e "${CYAN}============================================${NC}"
                echo -e "${CYAN}  第 $chapter 章: $desc - $exe_name${NC}"
                echo -e "${CYAN}============================================${NC}"
                echo ""

                if run_single_example "$exe_path"; then
                    ((chapter_success++))
                    ((total_success++))
                else
                    ((chapter_failed++))
                    ((total_failed++))
                fi

                echo ""
                echo "-------------------------------------------"
                echo ""
            done <<< "$exes"

            echo -e "${CYAN}第 $chapter 章小计: ${GREEN}成功 $chapter_success${NC}, ${RED}失败 $chapter_failed${NC}"
            echo ""
        fi
    done

    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  $title - 结果汇总${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo ""
    echo -e "成功: ${GREEN}$total_success${NC}"
    echo -e "失败: ${RED}$total_failed${NC}"
    echo -e "跳过: ${YELLOW}$total_skipped${NC}"
}

# ============================================================================
# 运行所有示例
# ============================================================================
run_all() {
    run_range 1 30 "运行所有示例"
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    # 检查构建目录
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}构建目录不存在${NC}"
        echo -e "${YELLOW}请先编译项目:${NC}"
        echo -e "${YELLOW}  cd examples && mkdir -p build && cd build${NC}"
        echo -e "${YELLOW}  cmake .. && make -j\$(nproc)${NC}"
        echo ""
        exit 1
    fi

    case "${1:-}" in
        -h|--help)
            show_help
            ;;
        -l|--list)
            list_executables
            ;;
        --list-chapter)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}错误: 请指定章节号${NC}"
                exit 1
            fi
            list_chapter "$2"
            ;;
        -a|--all)
            run_all
            ;;
        --all-chapters)
            run_all
            ;;
        --chapter-all)
            if [ -z "${2:-}" ]; then
                echo -e "${RED}错误: 请指定章节号${NC}"
                exit 1
            fi
            run_chapter_all "$2"
            ;;
        --all-basic)
            run_range 1 11 "基础篇 (1-11)"
            ;;
        --all-intermediate)
            run_range 12 19 "进阶篇 (12-19)"
            ;;
        --all-advanced)
            run_range 20 25 "高级篇 (20-25)"
            ;;
        --all-expert)
            run_range 26 30 "专家篇 (26-30)"
            ;;
        *)
            if [ -z "${1:-}" ]; then
                show_help
                exit 1
            fi
            run_example "$@"
            ;;
    esac
}

main "$@"