void x86_cpu_list (FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

                  const char *optarg)

{

    unsigned char model = !strcmp("?model", optarg);

    unsigned char dump = !strcmp("?dump", optarg);

    unsigned char cpuid = !strcmp("?cpuid", optarg);

    x86_def_t *def;

    char buf[256];



    if (cpuid) {

        (*cpu_fprintf)(f, "Recognized CPUID flags:\n");

        listflags(buf, sizeof (buf), (uint32_t)~0, feature_name, 1);

        (*cpu_fprintf)(f, "  f_edx: %s\n", buf);

        listflags(buf, sizeof (buf), (uint32_t)~0, ext_feature_name, 1);

        (*cpu_fprintf)(f, "  f_ecx: %s\n", buf);

        listflags(buf, sizeof (buf), (uint32_t)~0, ext2_feature_name, 1);

        (*cpu_fprintf)(f, "  extf_edx: %s\n", buf);

        listflags(buf, sizeof (buf), (uint32_t)~0, ext3_feature_name, 1);

        (*cpu_fprintf)(f, "  extf_ecx: %s\n", buf);

        return;

    }

    for (def = x86_defs; def; def = def->next) {

        snprintf(buf, sizeof (buf), def->flags ? "[%s]": "%s", def->name);

        if (model || dump) {

            (*cpu_fprintf)(f, "x86 %16s  %-48s\n", buf, def->model_id);

        } else {

            (*cpu_fprintf)(f, "x86 %16s\n", buf);

        }

        if (dump) {

            memcpy(buf, &def->vendor1, sizeof (def->vendor1));

            memcpy(buf + 4, &def->vendor2, sizeof (def->vendor2));

            memcpy(buf + 8, &def->vendor3, sizeof (def->vendor3));

            buf[12] = '\0';

            (*cpu_fprintf)(f,

                "  family %d model %d stepping %d level %d xlevel 0x%x"

                " vendor \"%s\"\n",

                def->family, def->model, def->stepping, def->level,

                def->xlevel, buf);

            listflags(buf, sizeof (buf), def->features, feature_name, 0);

            (*cpu_fprintf)(f, "  feature_edx %08x (%s)\n", def->features,

                buf);

            listflags(buf, sizeof (buf), def->ext_features, ext_feature_name,

                0);

            (*cpu_fprintf)(f, "  feature_ecx %08x (%s)\n", def->ext_features,

                buf);

            listflags(buf, sizeof (buf), def->ext2_features, ext2_feature_name,

                0);

            (*cpu_fprintf)(f, "  extfeature_edx %08x (%s)\n",

                def->ext2_features, buf);

            listflags(buf, sizeof (buf), def->ext3_features, ext3_feature_name,

                0);

            (*cpu_fprintf)(f, "  extfeature_ecx %08x (%s)\n",

                def->ext3_features, buf);

            (*cpu_fprintf)(f, "\n");

        }

    }

    if (kvm_enabled()) {

        (*cpu_fprintf)(f, "x86 %16s\n", "[host]");

    }

}
