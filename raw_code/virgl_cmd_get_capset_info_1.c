static void virgl_cmd_get_capset_info(VirtIOGPU *g,

                                      struct virtio_gpu_ctrl_command *cmd)

{

    struct virtio_gpu_get_capset_info info;

    struct virtio_gpu_resp_capset_info resp;



    VIRTIO_GPU_FILL_CMD(info);




    if (info.capset_index == 0) {

        resp.capset_id = VIRTIO_GPU_CAPSET_VIRGL;

        virgl_renderer_get_cap_set(resp.capset_id,

                                   &resp.capset_max_version,

                                   &resp.capset_max_size);

    } else {

        resp.capset_max_version = 0;

        resp.capset_max_size = 0;

    }

    resp.hdr.type = VIRTIO_GPU_RESP_OK_CAPSET_INFO;

    virtio_gpu_ctrl_response(g, cmd, &resp.hdr, sizeof(resp));

}