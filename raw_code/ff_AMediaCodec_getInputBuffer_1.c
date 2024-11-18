uint8_t* ff_AMediaCodec_getInputBuffer(FFAMediaCodec* codec, size_t idx, size_t *out_size)

{

    uint8_t *ret = NULL;

    JNIEnv *env = NULL;



    jobject buffer = NULL;



    JNI_GET_ENV_OR_RETURN(env, codec, NULL);



    if (codec->has_get_i_o_buffer) {

        buffer = (*env)->CallObjectMethod(env, codec->object, codec->jfields.get_input_buffer_id, idx);

        if (ff_jni_exception_check(env, 1, codec) < 0) {

            goto fail;

        }

    } else {

        if (!codec->input_buffers) {

            codec->input_buffers = (*env)->CallObjectMethod(env, codec->object, codec->jfields.get_input_buffers_id);

            if (ff_jni_exception_check(env, 1, codec) < 0) {

                goto fail;

            }



            codec->input_buffers = (*env)->NewGlobalRef(env, codec->input_buffers);

            if (ff_jni_exception_check(env, 1, codec) < 0) {

                goto fail;

            }

        }



        buffer = (*env)->GetObjectArrayElement(env, codec->input_buffers, idx);

        if (ff_jni_exception_check(env, 1, codec) < 0) {

            goto fail;

        }

    }



    ret = (*env)->GetDirectBufferAddress(env, buffer);

    *out_size = (*env)->GetDirectBufferCapacity(env, buffer);

fail:

    if (buffer) {

        (*env)->DeleteLocalRef(env, buffer);

    }



    return ret;

}
