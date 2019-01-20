DIRWORK 	= src/

IMAGNAME 	= working_enviroment
CONTNAME 	= work_enviroment

DOCK		= docker
BUILD		= build
RUN			= run
ATTC 		= attach
STOP		= stop
DEL			= rm

BFLAGS 		= . -t $(IMAGNAME)
RBFLAGS 	= -it -d -P --name $(CONTNAME) -v $(CURDIR)/$(DIRWORK):/home/work $(IMAGNAME)

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

