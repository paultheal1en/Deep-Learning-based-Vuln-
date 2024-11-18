void put_no_rnd_h264_chroma_mc8_altivec(uint8_t * dst, uint8_t * src, int stride, int h, int x, int y) {

   DECLARE_ALIGNED_16(signed int, ABCD[4]) =

                        {((8 - x) * (8 - y)),

                             ((x) * (8 - y)),

                         ((8 - x) * (y)),

                             ((x) * (y))};

    register int i;

    vec_u8 fperm;

    const vec_s32 vABCD = vec_ld(0, ABCD);

    const vec_s16 vA = vec_splat((vec_s16)vABCD, 1);

    const vec_s16 vB = vec_splat((vec_s16)vABCD, 3);

    const vec_s16 vC = vec_splat((vec_s16)vABCD, 5);

    const vec_s16 vD = vec_splat((vec_s16)vABCD, 7);

    LOAD_ZERO;

    const vec_s16 v28ss = vec_sub(vec_sl(vec_splat_s16(1),vec_splat_u16(5)),vec_splat_s16(4));

    const vec_u16 v6us  = vec_splat_u16(6);

    register int loadSecond     = (((unsigned long)src) % 16) <= 7 ? 0 : 1;

    register int reallyBadAlign = (((unsigned long)src) % 16) == 15 ? 1 : 0;



    vec_u8 vsrcAuc, vsrcBuc, vsrcperm0, vsrcperm1;

    vec_u8 vsrc0uc, vsrc1uc;

    vec_s16 vsrc0ssH, vsrc1ssH;

    vec_u8 vsrcCuc, vsrc2uc, vsrc3uc;

    vec_s16 vsrc2ssH, vsrc3ssH, psum;

    vec_u8 vdst, ppsum, fsum;



    if (((unsigned long)dst) % 16 == 0) {

        fperm = (vec_u8){0x10, 0x11, 0x12, 0x13,

                           0x14, 0x15, 0x16, 0x17,

                           0x08, 0x09, 0x0A, 0x0B,

                           0x0C, 0x0D, 0x0E, 0x0F};

    } else {

        fperm = (vec_u8){0x00, 0x01, 0x02, 0x03,

                           0x04, 0x05, 0x06, 0x07,

                           0x18, 0x19, 0x1A, 0x1B,

                           0x1C, 0x1D, 0x1E, 0x1F};

    }



    vsrcAuc = vec_ld(0, src);



    if (loadSecond)

        vsrcBuc = vec_ld(16, src);

    vsrcperm0 = vec_lvsl(0, src);

    vsrcperm1 = vec_lvsl(1, src);



    vsrc0uc = vec_perm(vsrcAuc, vsrcBuc, vsrcperm0);

    if (reallyBadAlign)

        vsrc1uc = vsrcBuc;

    else

        vsrc1uc = vec_perm(vsrcAuc, vsrcBuc, vsrcperm1);



    vsrc0ssH = (vec_s16)vec_mergeh(zero_u8v, (vec_u8)vsrc0uc);

    vsrc1ssH = (vec_s16)vec_mergeh(zero_u8v, (vec_u8)vsrc1uc);



    if (!loadSecond) {// -> !reallyBadAlign

        for (i = 0 ; i < h ; i++) {





            vsrcCuc = vec_ld(stride + 0, src);



            vsrc2uc = vec_perm(vsrcCuc, vsrcCuc, vsrcperm0);

            vsrc3uc = vec_perm(vsrcCuc, vsrcCuc, vsrcperm1);



            vsrc2ssH = (vec_s16)vec_mergeh(zero_u8v, (vec_u8)vsrc2uc);

            vsrc3ssH = (vec_s16)vec_mergeh(zero_u8v, (vec_u8)vsrc3uc);



            psum = vec_mladd(vA, vsrc0ssH, vec_splat_s16(0));

            psum = vec_mladd(vB, vsrc1ssH, psum);

            psum = vec_mladd(vC, vsrc2ssH, psum);

            psum = vec_mladd(vD, vsrc3ssH, psum);

            psum = vec_add(v28ss, psum);

            psum = vec_sra(psum, v6us);



            vdst = vec_ld(0, dst);

            ppsum = (vec_u8)vec_packsu(psum, psum);

            fsum = vec_perm(vdst, ppsum, fperm);



            vec_st(fsum, 0, dst);



            vsrc0ssH = vsrc2ssH;

            vsrc1ssH = vsrc3ssH;



            dst += stride;

            src += stride;

        }

    } else {

        vec_u8 vsrcDuc;

        for (i = 0 ; i < h ; i++) {

            vsrcCuc = vec_ld(stride + 0, src);

            vsrcDuc = vec_ld(stride + 16, src);



            vsrc2uc = vec_perm(vsrcCuc, vsrcDuc, vsrcperm0);

            if (reallyBadAlign)

                vsrc3uc = vsrcDuc;

            else

                vsrc3uc = vec_perm(vsrcCuc, vsrcDuc, vsrcperm1);



            vsrc2ssH = (vec_s16)vec_mergeh(zero_u8v, (vec_u8)vsrc2uc);

            vsrc3ssH = (vec_s16)vec_mergeh(zero_u8v, (vec_u8)vsrc3uc);



            psum = vec_mladd(vA, vsrc0ssH, vec_splat_s16(0));

            psum = vec_mladd(vB, vsrc1ssH, psum);

            psum = vec_mladd(vC, vsrc2ssH, psum);

            psum = vec_mladd(vD, vsrc3ssH, psum);

            psum = vec_add(v28ss, psum);

            psum = vec_sr(psum, v6us);



            vdst = vec_ld(0, dst);

            ppsum = (vec_u8)vec_pack(psum, psum);

            fsum = vec_perm(vdst, ppsum, fperm);



            vec_st(fsum, 0, dst);



            vsrc0ssH = vsrc2ssH;

            vsrc1ssH = vsrc3ssH;



            dst += stride;

            src += stride;

        }

    }

}
