#ifndef SM120_BLOCKSCALED_GEMM_SM120_SF_LAYOUT_H
#define SM120_BLOCKSCALED_GEMM_SM120_SF_LAYOUT_H

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

using namespace cute;

// Temporary adhoc partitioning for scaling factors.
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr
auto
thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
{
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr
auto
thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
{
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<1>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);                 // (PermN,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
    return thr_tensor;
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
{
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA =  thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFA);
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
{
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB =  thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFB);
}

template<class TiledMma>
CUTE_HOST_DEVICE constexpr
auto
get_layoutSFA_TV(TiledMma& mma)
{
    // (M,K) -> (M,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto atile = make_tile(_,
                           make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                 make_stride(               Int<1>{} ,                Int<0>{} )),
                                     _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

template<class TiledMma>
CUTE_HOST_DEVICE constexpr
auto
get_layoutSFB_TV(TiledMma& mma)
{
    // (N,K) -> (N,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto btile = make_tile(_,
                           make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                 make_stride(               Int<0>{} ,                Int<1>{} )),
                                     _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

template< int BM, int BN, int BK, int SFVecSize>
CUTE_HOST_DEVICE
auto
sm120_get_SF_layout(int M, int N, int K)
{
    static_assert(SFVecSize == 32, "SFVecSize should always be 32");
    static_assert(BK == 128, "BK should always be 128");
    static_assert(BM == BN, "BM and BN should always be same");
    using SfKMajorAtom  = Layout< Shape< Shape<Int<BM/4>,_4>, Shape<Int<SFVecSize>, _4>>,
            Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
    auto layout_SFA = tile_to_shape(SfKMajorAtom{}, make_shape(M,K), Step<_2,_1>{});
    auto layout_SFB = tile_to_shape(SfKMajorAtom{}, make_shape(N,K), Step<_2,_1>{});
    return make_tuple(layout_SFA, layout_SFB);
}

template<int BMN>
CUTE_HOST_DEVICE
auto
sm120_get_SFA_layout(int M, int K)
{
    constexpr int BK = 128;
    constexpr int SFVecSize = 32;
    // static_assert(SFVecSize == 32, "SFVecSize should always be 32");
    // static_assert(BK == 128, "BK should always be 128");
    // static_assert(BM == BN, "BM and BN should always be same");
    using SfKMajorAtom  = Layout< Shape< Shape<Int<BMN/4>,_4>, Shape<Int<SFVecSize>, _4>>,
            Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
    auto layout_SFA = tile_to_shape(SfKMajorAtom{}, make_shape(M,K), Step<_2,_1>{});
    // auto layout_SFB = tile_to_shape(SfKMajorAtom{}, make_shape(N,K), Step<_2,_1>{});
    return layout_SFA;
}

template<int BMN>
CUTE_HOST_DEVICE
auto
sm120_get_SFB_layout(int N, int K)
{
    constexpr int BK = 128;
    constexpr int SFVecSize = 32;
    // static_assert(SFVecSize == 32, "SFVecSize should always be 32");
    // static_assert(BK == 128, "BK should always be 128");
    // static_assert(BM == BN, "BM and BN should always be same");
    using SfKMajorAtom  = Layout< Shape< Shape<Int<BMN/4>,_4>, Shape<Int<SFVecSize>, _4>>,
            Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
    // auto layout_SFA = tile_to_shape(SfKMajorAtom{}, make_shape(M,K), Step<_2,_1>{});
    auto layout_SFB = tile_to_shape(SfKMajorAtom{}, make_shape(N,K), Step<_2,_1>{});
    return layout_SFB;
}
#endif //SM120_BLOCKSCALED_GEMM_SM120_SF_LAYOUT_H
