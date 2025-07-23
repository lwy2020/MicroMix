#include "fp4.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

void matmul_host4(
        const ElementA::DataType *A,
        const ElementB::DataType *B,
        int M,
        int N,
        int K,
        ElementC *C,
        ElementD *D,
        const ElementA::ScaleFactorType *SFA,
        const ElementB::ScaleFactorType *SFB
)
{
        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,                      
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
        >::CollectiveOp;
    
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
        >::CollectiveOp;
    
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,                                                   // Indicates ProblemShape
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;
    
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    
    // Reference device GEMM implementation type
    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    
    //
    // Data members
    //
    
    /// Initialization
    StrideA stride_A;
    LayoutSFA layout_SFA;
    StrideB stride_B;
    LayoutSFB layout_SFB;
    StrideC stride_C;
    StrideD stride_D;
    // For SFA and SFB tensors layouts
    using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    /***************************************** ↓ When performing benchmark, please comment out these lines ↓ *****************************************/
    // cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
    // cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
    // block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    // block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

    // int num_sfa_elements = M * (K / 32), num_sfb_elements = N * (K / 32);
    // std::vector<ElementA::ScaleFactorType> sfa_host_buffer(num_sfa_elements);
    // cudaMemcpy(sfa_host_buffer.data(), SFA , num_sfa_elements * sizeof(ElementA::ScaleFactorType), cudaMemcpyDeviceToHost);
    // std::vector<ElementB::ScaleFactorType> sfb_host_buffer(num_sfb_elements);
    // cudaMemcpy(sfb_host_buffer.data(), SFB , num_sfb_elements * sizeof(ElementB::ScaleFactorType), cudaMemcpyDeviceToHost);

    // ElementA::ScaleFactorType* host_sfa_data = block_SFA.host_data(); // Get pointer to HostTensor's data
    // ElementB::ScaleFactorType* host_sfb_data = block_SFB.host_data(); // Get pointer to HostTensor's data
    // auto sfa_tensor_on_host = cute::make_tensor(host_sfa_data, filter_zeros(layout_SFA));
    // auto sfb_tensor_on_host = cute::make_tensor(host_sfb_data, filter_zeros(layout_SFB));
    // // print(filter_zeros(layout_SFA).shape());
    // // print(filter_zeros(layout_SFB).shape());


    // for (int m_tile = 0; m_tile < M; ++m_tile) {
    //     for (int k_tile = 0; k_tile < K / 32; ++k_tile) {
    //         // Create a CUTE coordinate for the logical tile
    //         // The exact structure of this coord depends on how layout_SFA is defined.
    //         // For instance, if layout_SFA's shape is (M_TILES_SFA, K_TILES_SFA)
    //         int idx = m_tile * (K / 32) + k_tile;
    //         auto logical_coord0 = make_coord(make_coord(m_tile % 32, (m_tile / 32) % 4), m_tile / 128);
            
    //         auto logical_coord1 = make_coord(make_coord(0, k_tile % 4), k_tile / 4);

    //         auto logical_coord2 = make_coord(0, 0);
    //         // Get your pre-computed scale for this logical (m_tile, k_tile)
    //         ElementA::ScaleFactorType my_scale_value = sfa_host_buffer[m_tile * (K / 32) + k_tile];
    //         // Assign it to the tensor. CUTE handles the mapping to the 1D buffer.
    //         sfa_tensor_on_host(make_coord(logical_coord0, logical_coord1, logical_coord2)) = my_scale_value;
    //     }
    // }
    // for (int n_tile = 0; n_tile < N; ++n_tile) {
    //     for (int k_tile = 0; k_tile < K / 32; ++k_tile) {
    //         int idx = n_tile * (K / 32) + k_tile;
    //         auto logical_coord0 = make_coord(make_coord(n_tile % 32, (n_tile / 32) % 4), n_tile / 128);
            
    //         auto logical_coord1 = make_coord(make_coord(0, k_tile % 4), k_tile / 4);

    //         auto logical_coord2 = make_coord(0, 0);
    //         // Get your pre-computed scale for this logical (m_tile, k_tile)
    //         ElementB::ScaleFactorType my_scale_value = sfb_host_buffer[n_tile * (K / 32) + k_tile];
    //         // Assign it to the tensor. CUTE handles the mapping to the 1D buffer.
    //         sfb_tensor_on_host(make_coord(logical_coord0, logical_coord1, logical_coord2)) = my_scale_value;
    //     }
    // }
    
    // block_SFA.sync_device(); // Copy to GPU
    // block_SFB.sync_device(); // Copy to GPU
    /***************************************** ↑ When performing benchmark, please comment out these lines ↑ *****************************************/
    // Timing using CUDA events
    // cudaEvent_t start, stop;
    // CHECK_CUDA(cudaEventCreate(&start));
    // CHECK_CUDA(cudaEventCreate(&stop));
    // CHECK_CUDA(cudaEventRecord(start));
    Gemm gemmOp;

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
            A, stride_A,
            B, stride_B,
            SFA, layout_SFA,  //When performing benchmark, please repalce it with "SFA, layout_SFA,"
            SFB, layout_SFB   //When performing benchmark, please repalce it with "SFB, layout_SFB"
        },
        { // Epilogue arguments
            {1.0, 0},
            C, stride_C,
            D, stride_D
        }
    };

    auto status = gemmOp(arguments);
    if (status != cutlass::Status::kSuccess) {
        // 打印错误信息
        std::cerr << "CUTLASS GEMM operation in matmul_host4 failed with status: "
                  << cutlass::cutlassGetStatusString(status) // 使用 CUTLASS 提供的函数转换状态为字符串
                  << " (Enum value: " << static_cast<int>(status) << ")"
                  << std::endl;
    }
    assert(status == cutlass::Status::kSuccess);

}

