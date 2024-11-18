static void print_features(FILE *f,

                           int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

                           uint32_t features, const char *prefix)

{

    unsigned int i;



    for (i = 0; i < ARRAY_SIZE(feature_name); i++)

        if (feature_name[i] && (features & (1 << i))) {

            if (prefix)

                (*cpu_fprintf)(f, "%s", prefix);

            (*cpu_fprintf)(f, "%s ", feature_name[i]);

        }

}
