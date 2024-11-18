static void gen_check_sr(DisasContext *dc, uint32_t sr)

{

    if (!xtensa_option_bits_enabled(dc->config, sregnames[sr].opt_bits)) {

        if (sregnames[sr].name) {

            qemu_log("SR %s is not configured\n", sregnames[sr].name);

        } else {

            qemu_log("SR %d is not implemented\n", sr);

        }

        gen_exception_cause(dc, ILLEGAL_INSTRUCTION_CAUSE);

    }

}
