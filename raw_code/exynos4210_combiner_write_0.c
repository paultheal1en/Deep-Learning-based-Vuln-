static void exynos4210_combiner_write(void *opaque, target_phys_addr_t offset,

        uint64_t val, unsigned size)

{

    struct Exynos4210CombinerState *s =

            (struct Exynos4210CombinerState *)opaque;

    uint32_t req_quad_base_n;    /* Base of registers quad. Multiply it by 4 and

                                   get a start of corresponding group quad */

    uint32_t grp_quad_base_n;    /* Base of group quad */

    uint32_t reg_n;              /* Register number inside the quad */



    req_quad_base_n = offset >> 4;

    grp_quad_base_n = req_quad_base_n << 2;

    reg_n = (offset - (req_quad_base_n << 4)) >> 2;



    if (req_quad_base_n >= IIC_NGRP) {

        hw_error("exynos4210.combiner: unallowed write access at offset 0x"

                TARGET_FMT_plx "\n", offset);

        return;

    }



    if (reg_n > 1) {

        hw_error("exynos4210.combiner: unallowed write access at offset 0x"

                TARGET_FMT_plx "\n", offset);

        return;

    }



    if (offset >> 2 >= IIC_REGSET_SIZE) {

        hw_error("exynos4210.combiner: overflow of reg_set by 0x"

                TARGET_FMT_plx "offset\n", offset);

    }

    s->reg_set[offset >> 2] = val;



    switch (reg_n) {

    /* IIESR */

    case 0:

        /* FIXME: what if irq is pending, allowed by mask, and we allow it

         * again. Interrupt will rise again! */



        DPRINTF("%s enable IRQ for groups %d, %d, %d, %d\n",

                s->external ? "EXT" : "INT",

                grp_quad_base_n,

                grp_quad_base_n + 1,

                grp_quad_base_n + 2,

                grp_quad_base_n + 3);



        /* Enable interrupt sources */

        s->group[grp_quad_base_n].src_mask |= val & 0xFF;

        s->group[grp_quad_base_n + 1].src_mask |= (val & 0xFF00) >> 8;

        s->group[grp_quad_base_n + 2].src_mask |= (val & 0xFF0000) >> 16;

        s->group[grp_quad_base_n + 3].src_mask |= (val & 0xFF000000) >> 24;



        exynos4210_combiner_update(s, grp_quad_base_n);

        exynos4210_combiner_update(s, grp_quad_base_n + 1);

        exynos4210_combiner_update(s, grp_quad_base_n + 2);

        exynos4210_combiner_update(s, grp_quad_base_n + 3);

        break;

        /* IIECR */

    case 1:

        DPRINTF("%s disable IRQ for groups %d, %d, %d, %d\n",

                s->external ? "EXT" : "INT",

                grp_quad_base_n,

                grp_quad_base_n + 1,

                grp_quad_base_n + 2,

                grp_quad_base_n + 3);



        /* Disable interrupt sources */

        s->group[grp_quad_base_n].src_mask &= ~(val & 0xFF);

        s->group[grp_quad_base_n + 1].src_mask &= ~((val & 0xFF00) >> 8);

        s->group[grp_quad_base_n + 2].src_mask &= ~((val & 0xFF0000) >> 16);

        s->group[grp_quad_base_n + 3].src_mask &= ~((val & 0xFF000000) >> 24);



        exynos4210_combiner_update(s, grp_quad_base_n);

        exynos4210_combiner_update(s, grp_quad_base_n + 1);

        exynos4210_combiner_update(s, grp_quad_base_n + 2);

        exynos4210_combiner_update(s, grp_quad_base_n + 3);

        break;

    default:

        hw_error("exynos4210.combiner: unallowed write access at offset 0x"

                TARGET_FMT_plx "\n", offset);

        break;

    }

}
