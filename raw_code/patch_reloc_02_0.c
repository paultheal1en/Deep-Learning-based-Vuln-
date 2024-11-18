static void patch_reloc(uint8_t *code_ptr, int type,

                        tcg_target_long value, tcg_target_long addend)

{

    value += addend;

    switch (type) {

    case R_SPARC_32:

        if (value != (uint32_t)value)

            tcg_abort();

        *(uint32_t *)code_ptr = value;

        break;

    case R_SPARC_WDISP22:

        value -= (long)code_ptr;

        value >>= 2;

        if (!check_fit(value, 22))

            tcg_abort();

        *(uint32_t *)code_ptr = ((*(uint32_t *)code_ptr) & ~0x3fffff) | value;

        break;

    default:

        tcg_abort();

    }

}
