static int resample(ResampleContext *c, void *dst, const void *src,

                    int *consumed, int src_size, int dst_size, int update_ctx,

                    int nearest_neighbour)

{

    int dst_index;

    int index         = c->index;

    int frac          = c->frac;

    int dst_incr_frac = c->dst_incr % c->src_incr;

    int dst_incr      = c->dst_incr / c->src_incr;

    int compensation_distance = c->compensation_distance;



    if (!dst != !src)

        return AVERROR(EINVAL);



    if (nearest_neighbour) {

        int64_t index2 = ((int64_t)index) << 32;

        int64_t incr   = (1LL << 32) * c->dst_incr / c->src_incr;

        dst_size       = FFMIN(dst_size,

                               (src_size-1-index) * (int64_t)c->src_incr /

                               c->dst_incr);



        if (dst) {

            for(dst_index = 0; dst_index < dst_size; dst_index++) {

                c->resample_nearest(dst, dst_index, src, index2 >> 32);

                index2 += incr;

            }

        } else {

            dst_index = dst_size;

        }

        index += dst_index * dst_incr;

        index += (frac + dst_index * (int64_t)dst_incr_frac) / c->src_incr;

        frac   = (frac + dst_index * (int64_t)dst_incr_frac) % c->src_incr;

    } else {

        for (dst_index = 0; dst_index < dst_size; dst_index++) {

            int sample_index = index >> c->phase_shift;



            if (sample_index + c->filter_length > src_size ||

                -sample_index >= src_size)

                break;



            if (dst)

                c->resample_one(c, dst, dst_index, src, src_size, index, frac);



            frac  += dst_incr_frac;

            index += dst_incr;

            if (frac >= c->src_incr) {

                frac -= c->src_incr;

                index++;

            }

            if (dst_index + 1 == compensation_distance) {

                compensation_distance = 0;

                dst_incr_frac = c->ideal_dst_incr % c->src_incr;

                dst_incr      = c->ideal_dst_incr / c->src_incr;

            }

        }

    }

    if (consumed)

        *consumed = FFMAX(index, 0) >> c->phase_shift;



    if (update_ctx) {

        if (index >= 0)

            index &= c->phase_mask;



        if (compensation_distance) {

            compensation_distance -= dst_index;

            if (compensation_distance <= 0)

                return AVERROR_BUG;

        }

        c->frac     = frac;

        c->index    = index;

        c->dst_incr = dst_incr_frac + c->src_incr*dst_incr;

        c->compensation_distance = compensation_distance;

    }



    return dst_index;

}
