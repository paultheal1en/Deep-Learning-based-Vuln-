static int gen_neon_zip(int rd, int rm, int size, int q)

{

    TCGv tmp, tmp2;

    if (size == 3 || (!q && size == 2)) {

        return 1;

    }

    tmp = tcg_const_i32(rd);

    tmp2 = tcg_const_i32(rm);

    if (q) {

        switch (size) {

        case 0:

            gen_helper_neon_qzip8(tmp, tmp2);

            break;

        case 1:

            gen_helper_neon_qzip16(tmp, tmp2);

            break;

        case 2:

            gen_helper_neon_qzip32(tmp, tmp2);

            break;

        default:

            abort();

        }

    } else {

        switch (size) {

        case 0:

            gen_helper_neon_zip8(tmp, tmp2);

            break;

        case 1:

            gen_helper_neon_zip16(tmp, tmp2);

            break;

        default:

            abort();

        }

    }

    tcg_temp_free_i32(tmp);

    tcg_temp_free_i32(tmp2);

    return 0;

}
