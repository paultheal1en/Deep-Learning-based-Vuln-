exynos4210_combiner_read(void *opaque, target_phys_addr_t offset, unsigned size)

{

    struct Exynos4210CombinerState *s =

            (struct Exynos4210CombinerState *)opaque;

    uint32_t req_quad_base_n;    /* Base of registers quad. Multiply it by 4 and

                                   get a start of corresponding group quad */

    uint32_t grp_quad_base_n;    /* Base of group quad */

    uint32_t reg_n;              /* Register number inside the quad */

    uint32_t val;



    req_quad_base_n = offset >> 4;

    grp_quad_base_n = req_quad_base_n << 2;

    reg_n = (offset - (req_quad_base_n << 4)) >> 2;



    if (req_quad_base_n >= IIC_NGRP) {

        /* Read of ICIPSR register */

        return s->icipsr[reg_n];

    }



    val = 0;



    switch (reg_n) {

    /* IISTR */

    case 2:

        val |= s->group[grp_quad_base_n].src_pending;

        val |= s->group[grp_quad_base_n + 1].src_pending << 8;

        val |= s->group[grp_quad_base_n + 2].src_pending << 16;

        val |= s->group[grp_quad_base_n + 3].src_pending << 24;

        break;

    /* IIMSR */

    case 3:

        val |= s->group[grp_quad_base_n].src_mask &

        s->group[grp_quad_base_n].src_pending;

        val |= (s->group[grp_quad_base_n + 1].src_mask &

                s->group[grp_quad_base_n + 1].src_pending) << 8;

        val |= (s->group[grp_quad_base_n + 2].src_mask &

                s->group[grp_quad_base_n + 2].src_pending) << 16;

        val |= (s->group[grp_quad_base_n + 3].src_mask &

                s->group[grp_quad_base_n + 3].src_pending) << 24;

        break;

    default:

        if (offset >> 2 >= IIC_REGSET_SIZE) {

            hw_error("exynos4210.combiner: overflow of reg_set by 0x"

                    TARGET_FMT_plx "offset\n", offset);

        }

        val = s->reg_set[offset >> 2];

        return 0;

    }

    return val;

}
