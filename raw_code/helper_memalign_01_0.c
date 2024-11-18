void helper_memalign(uint32_t addr, uint32_t dr, uint32_t wr, uint32_t mask)

{

    if (addr & mask) {

            qemu_log("unaligned access addr=%x mask=%x, wr=%d\n",

                     addr, mask, wr);

            if (!(env->sregs[SR_MSR] & MSR_EE)) {

                return;

            }



            env->sregs[SR_ESR] = ESR_EC_UNALIGNED_DATA | (wr << 10) \

                                 | (dr & 31) << 5;

            if (mask == 3) {

                env->sregs[SR_ESR] |= 1 << 11;

            }

            helper_raise_exception(EXCP_HW_EXCP);

    }

}
