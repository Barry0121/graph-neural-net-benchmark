#########################################################################################################################
# Build with this command:                                                                                              #
#       docker build -t [barry121/graph-neural-network|barry121/graph-neural-network:<#version>] .                      #
# Note:                                                                                                                 #
# 1. choose one format for image name listed in the braket, first option uses the default version: "latest";            #
#    second option lets the user specify version name.                                                                  #
# 2. building process is going to take around 5-10 mins                                                                 #
#########################################################################################################################


# Start from ucsdets/scipy-ml-notebook
FROM ucsdets/scipy-ml-notebook

# Add everything to project folder
ADD . /dsc180-graph-neural-net
WORKDIR /dsc180-graph-neural-net

# OR Clone the Git Repo
# ARG username=${Barry0121}
# ARG password=${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}

# Install packages not in the original image
RUN conda create --name gcn --file ./environment.txt

# Activate the environment to use custom packages
# RUN conda activate gcn