void ff_schro_queue_free(FFSchroQueue *queue, void (*free_func)(void *))

{

    while (queue->p_head)

        free_func(ff_schro_queue_pop(queue));

}
