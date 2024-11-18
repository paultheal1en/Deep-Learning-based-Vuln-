static inline void gen_mtcr(CPUTriCoreState *env, DisasContext *ctx, TCGv r1,

                            int32_t offset)

{

    if (ctx->hflags & TRICORE_HFLAG_SM) {

        /* since we're caching PSW make this a special case */

        if (offset == 0xfe04) {

            gen_helper_psw_write(cpu_env, r1);

        } else {

            switch (offset) {

#include "csfr.def"

            }

        }

    } else {

        /* generate privilege trap */

    }

}
