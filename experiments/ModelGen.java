package neurnips20.experiments;

import ch.idsia.credici.IO;
import ch.idsia.credici.inference.CausalVE;
import ch.idsia.credici.inference.CredalCausalVE;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.credici.model.predefined.RandomChainNonMarkovian;
import ch.idsia.credici.model.predefined.RandomRevHMM;
import ch.idsia.credici.model.predefined.RandomSquares;
import ch.idsia.crema.factor.credal.vertex.VertexFactor;
import ch.idsia.crema.model.ObservationBuilder;
import ch.idsia.crema.utility.RandomUtil;
import gnu.trove.map.TIntIntMap;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ModelGen {

    static int endoVarSize = 2;
    static int exoVarSize = 5;


    public static void main(String[] args) throws InterruptedException {

        genModels("tree", 4);
        genModels("poly", 4);
        //genModels("multy", 4);

        genModels("tree", 6);
        genModels("poly", 6);
        //genModels("multy", 6);
    }


    public static void genModels(String model, int l){

        System.out.println(model);

        List<Long> seeds = new ArrayList<>();


        while(seeds.size() < 10) {
            long s1 = RandomUtil.sampleUniform(1, 50000, false)[0];
            long s2 = RandomUtil.sampleUniform(1, 50000, false)[0];
            //long s3 = RandomUtil.sampleUniform(1, 50000, false)[0];
            long s = s1+s2;
            RandomUtil.setRandomSeed(s); // imprecise causal result
            StructuralCausalModel causalModel;

            int n;
            int target;
            TIntIntMap intervention;
            TIntIntMap obs;


            if(model.equals("tree")) {
                n = l;
                causalModel = RandomChainNonMarkovian.buildModel(l, endoVarSize, exoVarSize);
                target = l-1;
                intervention = ObservationBuilder.observe(0, 1);
            }
            else if(model.equals("poly")) {
                n = l/2;
                causalModel = RandomRevHMM.buildModel(false, n, endoVarSize, exoVarSize);
                target = l-2;
                intervention = ObservationBuilder.observe(0, 1);

            }else { //"multy"
                n = l/2;
                causalModel = RandomSquares.buildModel(false, n, endoVarSize, exoVarSize);
                target = l-1;
                intervention = ObservationBuilder.observe(0, 1);
            }
            // query info
            System.out.println(s);
            try {
                CredalCausalVE inf = new CredalCausalVE(causalModel);
                VertexFactor res = (VertexFactor) inf.causalQuery()
                        .setTarget(target)
                        .setIntervention(intervention)
                        //.setEvidence(obs)
                        .run();

                System.out.println(res);
                if (res.getData()[0].length > 1 && !seeds.contains(s)) {
                    seeds.add(s);
                    IO.write(causalModel, "./experiments/models/"+model+""+l+"_"+s+".uai");
                }
            }catch(Exception e){
                System.out.println(e);
            }

        }

        System.out.println(seeds);

    }
}
