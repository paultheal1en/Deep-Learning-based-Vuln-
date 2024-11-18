void tb_target_set_jmp_target(uintptr_t tc_ptr, uintptr_t jmp_addr,

                              uintptr_t addr)

{

    if (TCG_TARGET_REG_BITS == 64) {

        tcg_insn_unit i1, i2;

        intptr_t tb_diff = addr - tc_ptr;

        intptr_t br_diff = addr - (jmp_addr + 4);

        uint64_t pair;



        /* This does not exercise the range of the branch, but we do

           still need to be able to load the new value of TCG_REG_TB.

           But this does still happen quite often.  */

        if (tb_diff == (int16_t)tb_diff) {

            i1 = ADDI | TAI(TCG_REG_TB, TCG_REG_TB, tb_diff);

            i2 = B | (br_diff & 0x3fffffc);

        } else {

            intptr_t lo = (int16_t)tb_diff;

            intptr_t hi = (int32_t)(tb_diff - lo);

            assert(tb_diff == hi + lo);

            i1 = ADDIS | TAI(TCG_REG_TB, TCG_REG_TB, hi >> 16);

            i2 = ADDI | TAI(TCG_REG_TB, TCG_REG_TB, lo);

        }

#ifdef HOST_WORDS_BIGENDIAN

        pair = (uint64_t)i1 << 32 | i2;

#else

        pair = (uint64_t)i2 << 32 | i1;

#endif



        atomic_set((uint64_t *)jmp_addr, pair);

        flush_icache_range(jmp_addr, jmp_addr + 8);

    } else {

        intptr_t diff = addr - jmp_addr;

        tcg_debug_assert(in_range_b(diff));

        atomic_set((uint32_t *)jmp_addr, B | (diff & 0x3fffffc));

        flush_icache_range(jmp_addr, jmp_addr + 4);

    }

}
