static void gen_neon_unzip(int reg, int q, int tmp, int size)

{

    int n;

    TCGv t0, t1;



    for (n = 0; n < q + 1; n += 2) {

        t0 = neon_load_reg(reg, n);

        t1 = neon_load_reg(reg, n + 1);

        switch (size) {

        case 0: gen_neon_unzip_u8(t0, t1); break;

        case 1: gen_neon_zip_u16(t0, t1); break; /* zip and unzip are the same.  */

        case 2: /* no-op */; break;

        default: abort();

        }

        neon_store_scratch(tmp + n, t0);

        neon_store_scratch(tmp + n + 1, t1);

    }

}
