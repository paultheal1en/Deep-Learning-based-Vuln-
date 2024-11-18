ARITH3(addlv)

ARITH3(sublv)

ARITH3(addqv)

ARITH3(subqv)

ARITH3(umulh)

ARITH3(mullv)

ARITH3(mulqv)

ARITH3(minub8)

ARITH3(minsb8)

ARITH3(minuw4)

ARITH3(minsw4)

ARITH3(maxub8)

ARITH3(maxsb8)

ARITH3(maxuw4)

ARITH3(maxsw4)

ARITH3(perr)



#define MVIOP2(name)                                    \

static inline void glue(gen_, name)(int rb, int rc)     \

{                                                       \

    if (unlikely(rc == 31))                             \

        return;                                         \

    if (unlikely(rb == 31))                             \

        tcg_gen_movi_i64(cpu_ir[rc], 0);                \

    else                                                \

        gen_helper_ ## name (cpu_ir[rc], cpu_ir[rb]);   \

}

MVIOP2(pklb)

MVIOP2(pkwb)

MVIOP2(unpkbl)

MVIOP2(unpkbw)



static void gen_cmp(TCGCond cond, int ra, int rb, int rc,

                    int islit, uint8_t lit)

{

    TCGv va, vb;



    if (unlikely(rc == 31)) {

        return;

    }



    if (ra == 31) {

        va = tcg_const_i64(0);

    } else {

        va = cpu_ir[ra];

    }

    if (islit) {

        vb = tcg_const_i64(lit);

    } else {

        vb = cpu_ir[rb];

    }



    tcg_gen_setcond_i64(cond, cpu_ir[rc], va, vb);



    if (ra == 31) {

        tcg_temp_free(va);

    }

    if (islit) {

        tcg_temp_free(vb);

    }

}
