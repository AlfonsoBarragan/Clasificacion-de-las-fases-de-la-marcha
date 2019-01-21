DIRWORK 	= src/

IMAGNAME 	= working_enviroment
CONTNAME 	= work_enviroment

DOCK		= docker
BUILD		= build
RUN			= run
ATTC 		= attach
STOP		= stop
DEL			= rm
PERM 		= chmod
SU 			= sudo

BFLAGS 		= . -t $(IMAGNAME)
RBFLAGS 	= -it -d -P --name $(CONTNAME) -v $(CURDIR)/$(DIRWORK):/root/work $(IMAGNAME)
PERMFLAGS 	= +777 

build:
	$(DOCK) $(BUILD) $(BFLAGS)

launch: bash work finish

work:
	$(DOCK) $(ATTC) $(CONTNAME)

bash:
	$(DOCK) $(RUN) $(RBFLAGS)

finish: stop delete

stop:
	$(DOCK) $(STOP) $(CONTNAME)

delete: 
	$(DOCK) $(DEL) $(CONTNAME)
	$(SU) $(PERM) $(PERMFLAGS) $(CURDIR)/$(DIRWORK)*

