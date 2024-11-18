static void x86_cpu_class_check_missing_features(X86CPUClass *xcc,

                                                 strList **missing_feats)

{

    X86CPU *xc;

    FeatureWord w;

    Error *err = NULL;

    strList **next = missing_feats;



    if (xcc->kvm_required && !kvm_enabled()) {

        strList *new = g_new0(strList, 1);

        new->value = g_strdup("kvm");;

        *missing_feats = new;

        return;

    }



    xc = X86_CPU(object_new(object_class_get_name(OBJECT_CLASS(xcc))));



    x86_cpu_load_features(xc, &err);

    if (err) {

        /* Errors at x86_cpu_load_features should never happen,

         * but in case it does, just report the model as not

         * runnable at all using the "type" property.

         */

        strList *new = g_new0(strList, 1);

        new->value = g_strdup("type");

        *next = new;

        next = &new->next;

    }



    x86_cpu_filter_features(xc);



    for (w = 0; w < FEATURE_WORDS; w++) {

        uint32_t filtered = xc->filtered_features[w];

        int i;

        for (i = 0; i < 32; i++) {

            if (filtered & (1UL << i)) {

                strList *new = g_new0(strList, 1);

                new->value = g_strdup(x86_cpu_feature_name(w, i));

                *next = new;

                next = &new->next;

            }

        }

    }



    object_unref(OBJECT(xc));

}
