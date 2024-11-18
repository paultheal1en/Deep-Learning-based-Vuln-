int add_exec(struct ex_list **ex_ptr, int do_pty, char *exec,

             struct in_addr addr, int port)

{

	struct ex_list *tmp_ptr;



	/* First, check if the port is "bound" */

	for (tmp_ptr = *ex_ptr; tmp_ptr; tmp_ptr = tmp_ptr->ex_next) {

		if (port == tmp_ptr->ex_fport &&

		    addr.s_addr == tmp_ptr->ex_addr.s_addr)

			return -1;

	}



	tmp_ptr = *ex_ptr;

	*ex_ptr = (struct ex_list *)malloc(sizeof(struct ex_list));

	(*ex_ptr)->ex_fport = port;

	(*ex_ptr)->ex_addr = addr;

	(*ex_ptr)->ex_pty = do_pty;

	(*ex_ptr)->ex_exec = (do_pty == 3) ? exec : strdup(exec);

	(*ex_ptr)->ex_next = tmp_ptr;

	return 0;

}
