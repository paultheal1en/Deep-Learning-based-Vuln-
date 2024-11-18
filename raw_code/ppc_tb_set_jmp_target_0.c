void ppc_tb_set_jmp_target(uintptr_t jmp_addr, uintptr_t addr)

{

    tcg_insn_unit i1, i2;

    uint64_t pair;

    intptr_t diff = addr - jmp_addr;



    if (in_range_b(diff)) {

        i1 = B | (diff & 0x3fffffc);

        i2 = NOP;

    } else if (USE_REG_RA) {

        intptr_t lo, hi;

        diff = addr - (uintptr_t)tb_ret_addr;

        lo = (int16_t)diff;

        hi = (int32_t)(diff - lo);

        assert(diff == hi + lo);

        i1 = ADDIS | TAI(TCG_REG_TMP1, TCG_REG_RA, hi >> 16);

        i2 = ADDI | TAI(TCG_REG_TMP1, TCG_REG_TMP1, lo);

    } else {

        assert(TCG_TARGET_REG_BITS == 32 || addr == (int32_t)addr);

        i1 = ADDIS | TAI(TCG_REG_TMP1, 0, addr >> 16);

        i2 = ORI | SAI(TCG_REG_TMP1, TCG_REG_TMP1, addr);

    }

#ifdef HOST_WORDS_BIGENDIAN

    pair = (uint64_t)i1 << 32 | i2;

#else

    pair = (uint64_t)i2 << 32 | i1;

#endif



    /* ??? __atomic_store_8, presuming there's some way to do that

       for 32-bit, otherwise this is good enough for 64-bit.  */

    *(uint64_t *)jmp_addr = pair;

    flush_icache_range(jmp_addr, jmp_addr + 8);

}
