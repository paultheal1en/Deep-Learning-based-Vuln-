static void handle_input(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIORNG *vrng = DO_UPCAST(VirtIORNG, vdev, vdev);

    size_t size;



    size = pop_an_elem(vrng);

    if (size) {

        rng_backend_request_entropy(vrng->rng, size, chr_read, vrng);

    }

}
