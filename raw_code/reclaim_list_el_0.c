static void reclaim_list_el(struct rcu_head *prcu)

{

    struct list_element *el = container_of(prcu, struct list_element, rcu);

    g_free(el);

    atomic_add(&n_reclaims, 1);

}
