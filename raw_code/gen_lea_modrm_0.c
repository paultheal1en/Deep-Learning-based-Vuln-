static void gen_lea_modrm(DisasContext *s, int modrm, int *reg_ptr, int *offset_ptr)

{

    int havesib;

    int base, disp;

    int index;

    int scale;

    int opreg;

    int mod, rm, code, override, must_add_seg;



    /* XXX: add a generation time variable to tell if base == 0 in DS/ES/SS */

    override = -1;

    must_add_seg = s->addseg;

    if (s->prefix & (PREFIX_CS | PREFIX_SS | PREFIX_DS | 

                     PREFIX_ES | PREFIX_FS | PREFIX_GS)) {

        if (s->prefix & PREFIX_ES)

            override = R_ES;

        else if (s->prefix & PREFIX_CS)

            override = R_CS;

        else if (s->prefix & PREFIX_SS)

            override = R_SS;

        else if (s->prefix & PREFIX_DS)

            override = R_DS;

        else if (s->prefix & PREFIX_FS)

            override = R_FS;

        else

            override = R_GS;

        must_add_seg = 1;

    }



    mod = (modrm >> 6) & 3;

    rm = modrm & 7;



    if (s->aflag) {



        havesib = 0;

        base = rm;

        index = 0;

        scale = 0;

        

        if (base == 4) {

            havesib = 1;

            code = ldub(s->pc++);

            scale = (code >> 6) & 3;

            index = (code >> 3) & 7;

            base = code & 7;

        }



        switch (mod) {

        case 0:

            if (base == 5) {

                base = -1;

                disp = ldl(s->pc);

                s->pc += 4;

            } else {

                disp = 0;

            }

            break;

        case 1:

            disp = (int8_t)ldub(s->pc++);

            break;

        default:

        case 2:

            disp = ldl(s->pc);

            s->pc += 4;

            break;

        }

        

        if (base >= 0) {

            gen_op_movl_A0_reg[base]();

            if (disp != 0)

                gen_op_addl_A0_im(disp);

        } else {

            gen_op_movl_A0_im(disp);

        }

        if (havesib && (index != 4 || scale != 0)) {

            gen_op_addl_A0_reg_sN[scale][index]();

        }

        if (must_add_seg) {

            if (override < 0) {

                if (base == R_EBP || base == R_ESP)

                    override = R_SS;

                else

                    override = R_DS;

            }

            gen_op_addl_A0_seg(offsetof(CPUX86State,seg_cache[override].base));

        }

    } else {

        switch (mod) {

        case 0:

            if (rm == 6) {

                disp = lduw(s->pc);

                s->pc += 2;

                gen_op_movl_A0_im(disp);

                rm = 0; /* avoid SS override */

                goto no_rm;

            } else {

                disp = 0;

            }

            break;

        case 1:

            disp = (int8_t)ldub(s->pc++);

            break;

        default:

        case 2:

            disp = lduw(s->pc);

            s->pc += 2;

            break;

        }

        switch(rm) {

        case 0:

            gen_op_movl_A0_reg[R_EBX]();

            gen_op_addl_A0_reg_sN[0][R_ESI]();

            break;

        case 1:

            gen_op_movl_A0_reg[R_EBX]();

            gen_op_addl_A0_reg_sN[0][R_EDI]();

            break;

        case 2:

            gen_op_movl_A0_reg[R_EBP]();

            gen_op_addl_A0_reg_sN[0][R_ESI]();

            break;

        case 3:

            gen_op_movl_A0_reg[R_EBP]();

            gen_op_addl_A0_reg_sN[0][R_EDI]();

            break;

        case 4:

            gen_op_movl_A0_reg[R_ESI]();

            break;

        case 5:

            gen_op_movl_A0_reg[R_EDI]();

            break;

        case 6:

            gen_op_movl_A0_reg[R_EBP]();

            break;

        default:

        case 7:

            gen_op_movl_A0_reg[R_EBX]();

            break;

        }

        if (disp != 0)

            gen_op_addl_A0_im(disp);

        gen_op_andl_A0_ffff();

    no_rm:

        if (must_add_seg) {

            if (override < 0) {

                if (rm == 2 || rm == 3 || rm == 6)

                    override = R_SS;

                else

                    override = R_DS;

            }

            gen_op_addl_A0_seg(offsetof(CPUX86State,seg_cache[override].base));

        }

    }



    opreg = OR_A0;

    disp = 0;

    *reg_ptr = opreg;

    *offset_ptr = disp;

}
