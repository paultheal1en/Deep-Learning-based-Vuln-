static void gen_rsr(DisasContext *dc, TCGv_i32 d, uint32_t sr)

{

    static void (* const rsr_handler[256])(DisasContext *dc,

            TCGv_i32 d, uint32_t sr) = {

        [CCOUNT] = gen_rsr_ccount,

        [PTEVADDR] = gen_rsr_ptevaddr,

    };



    if (sregnames[sr]) {

        if (rsr_handler[sr]) {

            rsr_handler[sr](dc, d, sr);

        } else {

            tcg_gen_mov_i32(d, cpu_SR[sr]);

        }

    } else {

        qemu_log("RSR %d not implemented, ", sr);

    }

}
