static TCGArg do_constant_folding_2(int op, TCGArg x, TCGArg y)

{

    switch (op) {

    CASE_OP_32_64(add):

        return x + y;



    CASE_OP_32_64(sub):

        return x - y;



    CASE_OP_32_64(mul):

        return x * y;



    CASE_OP_32_64(and):

        return x & y;



    CASE_OP_32_64(or):

        return x | y;



    CASE_OP_32_64(xor):

        return x ^ y;



    case INDEX_op_shl_i32:

        return (uint32_t)x << (uint32_t)y;



#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_shl_i64:

        return (uint64_t)x << (uint64_t)y;

#endif



    case INDEX_op_shr_i32:

        return (uint32_t)x >> (uint32_t)y;



#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_shr_i64:

        return (uint64_t)x >> (uint64_t)y;

#endif



    case INDEX_op_sar_i32:

        return (int32_t)x >> (int32_t)y;



#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_sar_i64:

        return (int64_t)x >> (int64_t)y;

#endif



#ifdef TCG_TARGET_HAS_rot_i32

    case INDEX_op_rotr_i32:

#if TCG_TARGET_REG_BITS == 64

        x &= 0xffffffff;

        y &= 0xffffffff;

#endif

        x = (x << (32 - y)) | (x >> y);

        return x;

#endif



#ifdef TCG_TARGET_HAS_rot_i64

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_rotr_i64:

        x = (x << (64 - y)) | (x >> y);

        return x;

#endif

#endif



#ifdef TCG_TARGET_HAS_rot_i32

    case INDEX_op_rotl_i32:

#if TCG_TARGET_REG_BITS == 64

        x &= 0xffffffff;

        y &= 0xffffffff;

#endif

        x = (x << y) | (x >> (32 - y));

        return x;

#endif



#ifdef TCG_TARGET_HAS_rot_i64

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_rotl_i64:

        x = (x << y) | (x >> (64 - y));

        return x;

#endif

#endif



#if defined(TCG_TARGET_HAS_not_i32) || defined(TCG_TARGET_HAS_not_i64)

#ifdef TCG_TARGET_HAS_not_i32

    case INDEX_op_not_i32:

#endif

#ifdef TCG_TARGET_HAS_not_i64

    case INDEX_op_not_i64:

#endif

        return ~x;

#endif



#if defined(TCG_TARGET_HAS_ext8s_i32) || defined(TCG_TARGET_HAS_ext8s_i64)

#ifdef TCG_TARGET_HAS_ext8s_i32

    case INDEX_op_ext8s_i32:

#endif

#ifdef TCG_TARGET_HAS_ext8s_i64

    case INDEX_op_ext8s_i64:

#endif

        return (int8_t)x;

#endif



#if defined(TCG_TARGET_HAS_ext16s_i32) || defined(TCG_TARGET_HAS_ext16s_i64)

#ifdef TCG_TARGET_HAS_ext16s_i32

    case INDEX_op_ext16s_i32:

#endif

#ifdef TCG_TARGET_HAS_ext16s_i64

    case INDEX_op_ext16s_i64:

#endif

        return (int16_t)x;

#endif



#if defined(TCG_TARGET_HAS_ext8u_i32) || defined(TCG_TARGET_HAS_ext8u_i64)

#ifdef TCG_TARGET_HAS_ext8u_i32

    case INDEX_op_ext8u_i32:

#endif

#ifdef TCG_TARGET_HAS_ext8u_i64

    case INDEX_op_ext8u_i64:

#endif

        return (uint8_t)x;

#endif



#if defined(TCG_TARGET_HAS_ext16u_i32) || defined(TCG_TARGET_HAS_ext16u_i64)

#ifdef TCG_TARGET_HAS_ext16u_i32

    case INDEX_op_ext16u_i32:

#endif

#ifdef TCG_TARGET_HAS_ext16u_i64

    case INDEX_op_ext16u_i64:

#endif

        return (uint16_t)x;

#endif



#if TCG_TARGET_REG_BITS == 64

#ifdef TCG_TARGET_HAS_ext32s_i64

    case INDEX_op_ext32s_i64:

        return (int32_t)x;

#endif



#ifdef TCG_TARGET_HAS_ext32u_i64

    case INDEX_op_ext32u_i64:

        return (uint32_t)x;

#endif

#endif



    default:

        fprintf(stderr,

                "Unrecognized operation %d in do_constant_folding.\n", op);

        tcg_abort();

    }

}
