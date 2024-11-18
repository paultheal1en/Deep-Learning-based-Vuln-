static ImageInfoSpecific *qcow2_get_specific_info(BlockDriverState *bs)

{

    BDRVQcow2State *s = bs->opaque;

    ImageInfoSpecific *spec_info = g_new(ImageInfoSpecific, 1);



    *spec_info = (ImageInfoSpecific){

        .type  = IMAGE_INFO_SPECIFIC_KIND_QCOW2,

        .u.qcow2.data = g_new(ImageInfoSpecificQCow2, 1),

    };

    if (s->qcow_version == 2) {

        *spec_info->u.qcow2.data = (ImageInfoSpecificQCow2){

            .compat             = g_strdup("0.10"),

            .refcount_bits      = s->refcount_bits,

        };

    } else if (s->qcow_version == 3) {

        *spec_info->u.qcow2.data = (ImageInfoSpecificQCow2){

            .compat             = g_strdup("1.1"),

            .lazy_refcounts     = s->compatible_features &

                                  QCOW2_COMPAT_LAZY_REFCOUNTS,

            .has_lazy_refcounts = true,

            .corrupt            = s->incompatible_features &

                                  QCOW2_INCOMPAT_CORRUPT,

            .has_corrupt        = true,

            .refcount_bits      = s->refcount_bits,

        };

    } else {

        /* if this assertion fails, this probably means a new version was

         * added without having it covered here */

        assert(false);

    }



    return spec_info;

}
