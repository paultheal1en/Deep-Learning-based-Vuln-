static void virtio_input_handle_event(DeviceState *dev, QemuConsole *src,

                                      InputEvent *evt)

{

    VirtIOInput *vinput = VIRTIO_INPUT(dev);

    virtio_input_event event;

    int qcode;



    switch (evt->kind) {

    case INPUT_EVENT_KIND_KEY:

        qcode = qemu_input_key_value_to_qcode(evt->key->key);

        if (qcode && keymap_qcode[qcode]) {

            event.type  = cpu_to_le16(EV_KEY);

            event.code  = cpu_to_le16(keymap_qcode[qcode]);

            event.value = cpu_to_le32(evt->key->down ? 1 : 0);

            virtio_input_send(vinput, &event);

        } else {

            if (evt->key->down) {

                fprintf(stderr, "%s: unmapped key: %d [%s]\n", __func__,

                        qcode, QKeyCode_lookup[qcode]);

            }

        }

        break;

    case INPUT_EVENT_KIND_BTN:

        if (keymap_button[evt->btn->button]) {

            event.type  = cpu_to_le16(EV_KEY);

            event.code  = cpu_to_le16(keymap_button[evt->btn->button]);

            event.value = cpu_to_le32(evt->btn->down ? 1 : 0);

            virtio_input_send(vinput, &event);

        } else {

            if (evt->btn->down) {

                fprintf(stderr, "%s: unmapped button: %d [%s]\n", __func__,

                        evt->btn->button, InputButton_lookup[evt->btn->button]);

            }

        }

        break;

    case INPUT_EVENT_KIND_REL:

        event.type  = cpu_to_le16(EV_REL);

        event.code  = cpu_to_le16(axismap_rel[evt->rel->axis]);

        event.value = cpu_to_le32(evt->rel->value);

        virtio_input_send(vinput, &event);

        break;

    case INPUT_EVENT_KIND_ABS:

        event.type  = cpu_to_le16(EV_ABS);

        event.code  = cpu_to_le16(axismap_abs[evt->abs->axis]);

        event.value = cpu_to_le32(evt->abs->value);

        virtio_input_send(vinput, &event);

        break;

    default:

        /* keep gcc happy */

        break;

    }

}
