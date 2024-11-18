static void usb_mtp_command(MTPState *s, MTPControl *c)

{

    MTPData *data_in = NULL;

    MTPObject *o;

    uint32_t nres = 0, res0 = 0;



    /* sanity checks */

    if (c->code >= CMD_CLOSE_SESSION && s->session == 0) {

        usb_mtp_queue_result(s, RES_SESSION_NOT_OPEN,

                             c->trans, 0, 0, 0);

        return;

    }



    /* process commands */

    switch (c->code) {

    case CMD_GET_DEVICE_INFO:

        data_in = usb_mtp_get_device_info(s, c);

        break;

    case CMD_OPEN_SESSION:

        if (s->session) {

            usb_mtp_queue_result(s, RES_SESSION_ALREADY_OPEN,

                                 c->trans, 1, s->session, 0);

            return;

        }

        if (c->argv[0] == 0) {

            usb_mtp_queue_result(s, RES_INVALID_PARAMETER,

                                 c->trans, 0, 0, 0);

            return;

        }

        trace_usb_mtp_op_open_session(s->dev.addr);

        s->session = c->argv[0];

        usb_mtp_object_alloc(s, s->next_handle++, NULL, s->root);

#ifdef __linux__

        if (usb_mtp_inotify_init(s)) {

            fprintf(stderr, "usb-mtp: file monitoring init failed\n");

        }

#endif

        break;

    case CMD_CLOSE_SESSION:

        trace_usb_mtp_op_close_session(s->dev.addr);

        s->session = 0;

        s->next_handle = 0;

#ifdef __linux__

        usb_mtp_inotify_cleanup(s);

#endif

        usb_mtp_object_free(s, QTAILQ_FIRST(&s->objects));

        assert(QTAILQ_EMPTY(&s->objects));

        break;

    case CMD_GET_STORAGE_IDS:

        data_in = usb_mtp_get_storage_ids(s, c);

        break;

    case CMD_GET_STORAGE_INFO:

        if (c->argv[0] != QEMU_STORAGE_ID &&

            c->argv[0] != 0xffffffff) {

            usb_mtp_queue_result(s, RES_INVALID_STORAGE_ID,

                                 c->trans, 0, 0, 0);

            return;

        }

        data_in = usb_mtp_get_storage_info(s, c);

        break;

    case CMD_GET_NUM_OBJECTS:

    case CMD_GET_OBJECT_HANDLES:

        if (c->argv[0] != QEMU_STORAGE_ID &&

            c->argv[0] != 0xffffffff) {

            usb_mtp_queue_result(s, RES_INVALID_STORAGE_ID,

                                 c->trans, 0, 0, 0);

            return;

        }

        if (c->argv[1] != 0x00000000) {

            usb_mtp_queue_result(s, RES_SPEC_BY_FORMAT_UNSUPPORTED,

                                 c->trans, 0, 0, 0);

            return;

        }

        if (c->argv[2] == 0x00000000 ||

            c->argv[2] == 0xffffffff) {

            o = QTAILQ_FIRST(&s->objects);

        } else {

            o = usb_mtp_object_lookup(s, c->argv[2]);

        }

        if (o == NULL) {

            usb_mtp_queue_result(s, RES_INVALID_OBJECT_HANDLE,

                                 c->trans, 0, 0, 0);

            return;

        }

        if (o->format != FMT_ASSOCIATION) {

            usb_mtp_queue_result(s, RES_INVALID_PARENT_OBJECT,

                                 c->trans, 0, 0, 0);

            return;

        }

        usb_mtp_object_readdir(s, o);

        if (c->code == CMD_GET_NUM_OBJECTS) {

            trace_usb_mtp_op_get_num_objects(s->dev.addr, o->handle, o->path);

            nres = 1;

            res0 = o->nchildren;

        } else {

            data_in = usb_mtp_get_object_handles(s, c, o);

        }

        break;

    case CMD_GET_OBJECT_INFO:

        o = usb_mtp_object_lookup(s, c->argv[0]);

        if (o == NULL) {

            usb_mtp_queue_result(s, RES_INVALID_OBJECT_HANDLE,

                                 c->trans, 0, 0, 0);

            return;

        }

        data_in = usb_mtp_get_object_info(s, c, o);

        break;

    case CMD_GET_OBJECT:

        o = usb_mtp_object_lookup(s, c->argv[0]);

        if (o == NULL) {

            usb_mtp_queue_result(s, RES_INVALID_OBJECT_HANDLE,

                                 c->trans, 0, 0, 0);

            return;

        }

        if (o->format == FMT_ASSOCIATION) {

            usb_mtp_queue_result(s, RES_INVALID_OBJECT_HANDLE,

                                 c->trans, 0, 0, 0);

            return;

        }

        data_in = usb_mtp_get_object(s, c, o);

        if (data_in == NULL) {

            usb_mtp_queue_result(s, RES_GENERAL_ERROR,

                                 c->trans, 0, 0, 0);

            return;

        }

        break;

    case CMD_GET_PARTIAL_OBJECT:

        o = usb_mtp_object_lookup(s, c->argv[0]);

        if (o == NULL) {

            usb_mtp_queue_result(s, RES_INVALID_OBJECT_HANDLE,

                                 c->trans, 0, 0, 0);

            return;

        }

        if (o->format == FMT_ASSOCIATION) {

            usb_mtp_queue_result(s, RES_INVALID_OBJECT_HANDLE,

                                 c->trans, 0, 0, 0);

            return;

        }

        data_in = usb_mtp_get_partial_object(s, c, o);

        if (data_in == NULL) {

            usb_mtp_queue_result(s, RES_GENERAL_ERROR,

                                 c->trans, 0, 0, 0);

            return;

        }

        nres = 1;

        res0 = data_in->length;

        break;

    default:

        trace_usb_mtp_op_unknown(s->dev.addr, c->code);

        usb_mtp_queue_result(s, RES_OPERATION_NOT_SUPPORTED,

                             c->trans, 0, 0, 0);

        return;

    }



    /* return results on success */

    if (data_in) {

        assert(s->data_in == NULL);

        s->data_in = data_in;

    }

    usb_mtp_queue_result(s, RES_OK, c->trans, nres, res0, 0);

}
