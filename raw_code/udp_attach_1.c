udp_attach(struct socket *so)

{

  if((so->s = socket(AF_INET,SOCK_DGRAM,0)) != -1) {

    so->so_expire = curtime + SO_EXPIRE;

    insque(so, &so->slirp->udb);

  }

  return(so->s);

}
