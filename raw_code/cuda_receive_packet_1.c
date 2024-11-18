static void cuda_receive_packet(CUDAState *s,

                                const uint8_t *data, int len)

{

    uint8_t obuf[16] = { CUDA_PACKET, 0, data[0] };

    int autopoll;

    uint32_t ti;



    switch(data[0]) {

    case CUDA_AUTOPOLL:

        autopoll = (data[1] != 0);

        if (autopoll != s->autopoll) {

            s->autopoll = autopoll;

            if (autopoll) {

                timer_mod(s->adb_poll_timer,

                               qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +

                               (get_ticks_per_sec() / CUDA_ADB_POLL_FREQ));

            } else {

                timer_del(s->adb_poll_timer);

            }

        }

        cuda_send_packet_to_host(s, obuf, 3);

        break;

    case CUDA_GET_6805_ADDR:

        cuda_send_packet_to_host(s, obuf, 3);

        break;

    case CUDA_SET_TIME:

        ti = (((uint32_t)data[1]) << 24) + (((uint32_t)data[2]) << 16) + (((uint32_t)data[3]) << 8) + data[4];

        s->tick_offset = ti - (qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) / get_ticks_per_sec());

        cuda_send_packet_to_host(s, obuf, 3);

        break;

    case CUDA_GET_TIME:

        ti = s->tick_offset + (qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) / get_ticks_per_sec());

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

        cuda_send_packet_to_host(s, obuf, 3);

        break;

    case CUDA_POWERDOWN:

        cuda_send_packet_to_host(s, obuf, 3);

        qemu_system_shutdown_request();

        break;

    case CUDA_RESET_SYSTEM:

        cuda_send_packet_to_host(s, obuf, 3);

        qemu_system_reset_request();

        break;

    case CUDA_COMBINED_FORMAT_IIC:


        obuf[1] = 0x5;




        break;

    case CUDA_GET_SET_IIC:

        if (len == 4) {

            cuda_send_packet_to_host(s, obuf, 3);

        } else {






        }

        break;

    default:






        break;

    }

}