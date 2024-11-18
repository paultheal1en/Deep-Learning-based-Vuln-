target_ulong helper_srad(CPUPPCState *env, target_ulong value,

                         target_ulong shift)

{

    int64_t ret;



    if (likely(!(shift & 0x40))) {

        if (likely((uint64_t)shift != 0)) {

            shift &= 0x3f;

            ret = (int64_t)value >> shift;

            if (likely(ret >= 0 || (value & ((1 << shift) - 1)) == 0)) {

                env->ca = 0;

            } else {

                env->ca = 1;

            }

        } else {

            ret = (int64_t)value;

            env->ca = 0;

        }

    } else {

        ret = (int64_t)value >> 63;

        env->ca = (ret != 0);

    }

    return ret;

}
