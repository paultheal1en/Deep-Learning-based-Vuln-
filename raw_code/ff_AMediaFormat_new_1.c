FFAMediaFormat *ff_AMediaFormat_new(void)

{

    JNIEnv *env = NULL;

    FFAMediaFormat *format = NULL;



    format = av_mallocz(sizeof(FFAMediaFormat));

    if (!format) {

        return NULL;

    }

    format->class = &amediaformat_class;



    env = ff_jni_get_env(format);

    if (!env) {

        av_freep(&format);

        return NULL;

    }



    if (ff_jni_init_jfields(env, &format->jfields, jni_amediaformat_mapping, 1, format) < 0) {

        goto fail;

    }



    format->object = (*env)->NewObject(env, format->jfields.mediaformat_class, format->jfields.init_id);

    if (!format->object) {

        goto fail;

    }



    format->object = (*env)->NewGlobalRef(env, format->object);

    if (!format->object) {

        goto fail;

    }



    return format;

fail:

    ff_jni_reset_jfields(env, &format->jfields, jni_amediaformat_mapping, 1, format);



    av_freep(&format);



    return NULL;

}
