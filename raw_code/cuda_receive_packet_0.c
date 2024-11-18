static void cuda_receive_packet(CUDAState *s,

                                const uint8_t *data, int len)

{

    uint8_t obuf[16];

    int ti, autopoll;



    switch(data[0]) {

    case CUDA_AUTOPOLL:

        autopoll = (data[1] != 0);

        if (autopoll != s->autopoll) {

            s->autopoll = autopoll;

            if (autopoll) {

                qemu_mod_timer(s->adb_poll_timer,

                               qemu_get_clock(vm_clock) +

                               (ticks_per_sec / CUDA_ADB_POLL_FREQ));

            } else {

                qemu_del_timer(s->adb_poll_timer);

            }

        }

        obuf[0] = CUDA_PACKET;

        obuf[1] = data[1];

        cuda_send_packet_to_host(s, obuf, 2);

        break;

    case CUDA_GET_TIME:

    case CUDA_SET_TIME:

        /* XXX: add time support ? */

        ti = time(NULL) + RTC_OFFSET;

        obuf[0] = CUDA_PACKET;

        obuf[1] = 0;

        obuf[2] = 0;

        obuf[3] = ti >> 24;

        obuf[4] = ti >> 16;

        obuf[5] = ti >> 8;

        obuf[6] = ti;

        cuda_send_packet_to_host(s, obuf, 7);

        break;

    case CUDA_FILE_SERVER_FLAG:

    case CUDA_SET_DEVICE_LIST:

    case CUDA_SET_AUTO_RATE:

    case CUDA_SET_POWER_MESSAGES:

        obuf[0] = CUDA_PACKET;

        obuf[1] = 0;

        cuda_send_packet_to_host(s, obuf, 2);

        break;

    case CUDA_POWERDOWN:

        obuf[0] = CUDA_PACKET;

        obuf[1] = 0;

        cuda_send_packet_to_host(s, obuf, 2);

	qemu_system_shutdown_request();

	break;

    case CUDA_RESET_SYSTEM:

        obuf[0] = CUDA_PACKET;

        obuf[1] = 0;

        cuda_send_packet_to_host(s, obuf, 2);

        qemu_system_reset_request();

        break;

    default:

        break;

    }

}
