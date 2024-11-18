static mfxIMPL choose_implementation(const InputStream *ist)

{

    static const struct {

        const char *name;

        mfxIMPL     impl;

    } impl_map[] = {

        { "auto",     MFX_IMPL_AUTO         },

        { "sw",       MFX_IMPL_SOFTWARE     },

        { "hw",       MFX_IMPL_HARDWARE     },

        { "auto_any", MFX_IMPL_AUTO_ANY     },

        { "hw_any",   MFX_IMPL_HARDWARE_ANY },

        { "hw2",      MFX_IMPL_HARDWARE2    },

        { "hw3",      MFX_IMPL_HARDWARE3    },

        { "hw4",      MFX_IMPL_HARDWARE4    },

    };



    mfxIMPL impl = MFX_IMPL_AUTO_ANY;

    int i;



    if (ist->hwaccel_device) {

        for (i = 0; i < FF_ARRAY_ELEMS(impl_map); i++)

            if (!strcmp(ist->hwaccel_device, impl_map[i].name)) {

                impl = impl_map[i].impl;

                break;

            }

        if (i == FF_ARRAY_ELEMS(impl_map))

            impl = strtol(ist->hwaccel_device, NULL, 0);

    }



    return impl;

}
