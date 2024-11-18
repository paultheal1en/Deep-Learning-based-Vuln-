int av_parse_cpu_flags(const char *s)

{

#define CPUFLAG_MMXEXT   (AV_CPU_FLAG_MMX      | AV_CPU_FLAG_MMXEXT | AV_CPU_FLAG_CMOV)

#define CPUFLAG_3DNOW    (AV_CPU_FLAG_3DNOW    | AV_CPU_FLAG_MMX)

#define CPUFLAG_3DNOWEXT (AV_CPU_FLAG_3DNOWEXT | CPUFLAG_3DNOW)

#define CPUFLAG_SSE      (AV_CPU_FLAG_SSE      | CPUFLAG_MMXEXT)

#define CPUFLAG_SSE2     (AV_CPU_FLAG_SSE2     | CPUFLAG_SSE)

#define CPUFLAG_SSE2SLOW (AV_CPU_FLAG_SSE2SLOW | CPUFLAG_SSE2)

#define CPUFLAG_SSE3     (AV_CPU_FLAG_SSE3     | CPUFLAG_SSE2)

#define CPUFLAG_SSE3SLOW (AV_CPU_FLAG_SSE3SLOW | CPUFLAG_SSE3)

#define CPUFLAG_SSSE3    (AV_CPU_FLAG_SSSE3    | CPUFLAG_SSE3)

#define CPUFLAG_SSE4     (AV_CPU_FLAG_SSE4     | CPUFLAG_SSSE3)

#define CPUFLAG_SSE42    (AV_CPU_FLAG_SSE42    | CPUFLAG_SSE4)

#define CPUFLAG_AVX      (AV_CPU_FLAG_AVX      | CPUFLAG_SSE42)

#define CPUFLAG_AVXSLOW  (AV_CPU_FLAG_AVXSLOW  | CPUFLAG_AVX)

#define CPUFLAG_XOP      (AV_CPU_FLAG_XOP      | CPUFLAG_AVX)

#define CPUFLAG_FMA3     (AV_CPU_FLAG_FMA3     | CPUFLAG_AVX)

#define CPUFLAG_FMA4     (AV_CPU_FLAG_FMA4     | CPUFLAG_AVX)

#define CPUFLAG_AVX2     (AV_CPU_FLAG_AVX2     | CPUFLAG_AVX)

#define CPUFLAG_BMI2     (AV_CPU_FLAG_BMI2     | AV_CPU_FLAG_BMI1)

    static const AVOption cpuflags_opts[] = {

        { "flags"   , NULL, 0, AV_OPT_TYPE_FLAGS, { .i64 = 0 }, INT64_MIN, INT64_MAX, .unit = "flags" },

#if   ARCH_PPC

        { "altivec" , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_ALTIVEC  },    .unit = "flags" },

#elif ARCH_X86

        { "mmx"     , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_MMX      },    .unit = "flags" },

        { "mmxext"  , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_MMXEXT       },    .unit = "flags" },

        { "sse"     , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_SSE          },    .unit = "flags" },

        { "sse2"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_SSE2         },    .unit = "flags" },

        { "sse2slow", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_SSE2SLOW     },    .unit = "flags" },

        { "sse3"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_SSE3         },    .unit = "flags" },

        { "sse3slow", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_SSE3SLOW     },    .unit = "flags" },

        { "ssse3"   , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_SSSE3        },    .unit = "flags" },

        { "atom"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_ATOM     },    .unit = "flags" },

        { "sse4.1"  , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_SSE4         },    .unit = "flags" },

        { "sse4.2"  , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_SSE42        },    .unit = "flags" },

        { "avx"     , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_AVX          },    .unit = "flags" },

        { "avxslow" , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_AVXSLOW      },    .unit = "flags" },

        { "xop"     , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_XOP          },    .unit = "flags" },

        { "fma3"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_FMA3         },    .unit = "flags" },

        { "fma4"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_FMA4         },    .unit = "flags" },

        { "avx2"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_AVX2         },    .unit = "flags" },

        { "bmi1"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_BMI1     },    .unit = "flags" },

        { "bmi2"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_BMI2         },    .unit = "flags" },

        { "3dnow"   , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_3DNOW        },    .unit = "flags" },

        { "3dnowext", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CPUFLAG_3DNOWEXT     },    .unit = "flags" },

        { "cmov",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_CMOV     },    .unit = "flags" },

#elif ARCH_ARM

        { "armv5te",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_ARMV5TE  },    .unit = "flags" },

        { "armv6",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_ARMV6    },    .unit = "flags" },

        { "armv6t2",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_ARMV6T2  },    .unit = "flags" },

        { "vfp",      NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_VFP      },    .unit = "flags" },


        { "vfpv3",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_VFPV3    },    .unit = "flags" },

        { "neon",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_NEON     },    .unit = "flags" },

#elif ARCH_AARCH64

        { "armv8",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_ARMV8    },    .unit = "flags" },

        { "neon",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_NEON     },    .unit = "flags" },

        { "vfp",      NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_CPU_FLAG_VFP      },    .unit = "flags" },

#endif

        { NULL },

    };

    static const AVClass class = {

        .class_name = "cpuflags",

        .item_name  = av_default_item_name,

        .option     = cpuflags_opts,

        .version    = LIBAVUTIL_VERSION_INT,

    };



    int flags = 0, ret;

    const AVClass *pclass = &class;



    if ((ret = av_opt_eval_flags(&pclass, &cpuflags_opts[0], s, &flags)) < 0)

        return ret;



    return flags & INT_MAX;

}