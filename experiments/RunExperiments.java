package neurnips20.experiments;

import ch.idsia.credici.IO;
import ch.idsia.credici.inference.CausalVE;
import ch.idsia.credici.inference.CredalCausalApproxLP;
import ch.idsia.credici.inference.CredalCausalVE;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.crema.data.WriterCSV;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.factor.convert.BayesianToInterval;
import ch.idsia.crema.factor.convert.VertexToInterval;
import ch.idsia.crema.factor.credal.linear.IntervalFactor;
import ch.idsia.crema.factor.credal.vertex.VertexFactor;
import ch.idsia.crema.learning.ExpectationMaximization;
import ch.idsia.crema.model.ObservationBuilder;
import ch.idsia.crema.model.Strides;
import ch.idsia.crema.model.graphical.SparseModel;
import ch.idsia.crema.model.graphical.specialized.BayesianNetwork;
import ch.idsia.crema.user.credal.Vertex;
import com.google.common.primitives.Doubles;
import gnu.trove.map.TIntIntMap;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static java.util.Collections.*;

public class RunExperiments {


    static String modelName = "./models/scm2.uai";
    static String modelType = "";
    static StructuralCausalModel causalModel;
    static int numberPoints = 5;
    static int numberEMiter = 10 ;
    static int samples = 2000;
    static  TIntIntMap[] data;

    static int EMiter[];
    static double EMtime[];
    static double UQtime[];
    static double CNtime[];

    static int target;
    static TIntIntMap intervention;
    static TIntIntMap obs;

    static IntervalFactor[] ifactors;
    static BayesianNetwork[] bnets;
    static VertexFactor exactRes;

    static IntervalFactor ALPres;

    static Map<Integer, Map<String, Double>> results;

    static boolean filterNonConverging = false;

    public static void main(String[] args) {

        boolean error = false;
        try {
            parseArgs(args);
            init();
            run();
            buildResults();
            printResults();
        } catch (Exception e){
            e.printStackTrace();
            System.out.println(e.getMessage());
            error = true;
        } catch (Error e){
            e.printStackTrace();
            System.out.println(e.getMessage());
            error = true;
        }

        if(error){
            System.out.println("<output>[]</output>");
        }


    }

    public static void init() throws IOException {
        causalModel = (StructuralCausalModel) IO.read(modelName);
        if(modelName.contains("poly"))
            modelType = "poly";
        else if(modelName.contains("tree"))
            modelType = "tree";


        if(modelType.equals("poly"))
            target = causalModel.getEndogenousVars().length -2;
        else
            target = causalModel.getEndogenousVars().length - 1;

        intervention = ObservationBuilder.observe(0, 1);

        data =  IntStream.range(0,samples).mapToObj(i -> causalModel.sample(causalModel.getEndogenousVars())).toArray(TIntIntMap[]::new);

        results = new HashMap<Integer, Map<String, Double>>();

        for(int i=1; i<=numberPoints; i++){
            results.put(i, new HashMap<String, Double>());
        }

        ifactors = new IntervalFactor[numberPoints];
        bnets = new BayesianNetwork[numberPoints];
        EMiter = new int[numberPoints];

        EMtime = new double[numberPoints];
        UQtime = new double[numberPoints];
        CNtime = new double[numberPoints];


    }

    public static void run() throws InterruptedException {


        // Compute the exact results
        CredalCausalVE inf = new CredalCausalVE(causalModel);
        exactRes = (VertexFactor) inf.causalQuery().setTarget(target).setIntervention(intervention).run();

        CredalCausalApproxLP inf2 = new CredalCausalApproxLP(causalModel);
        ALPres = (IntervalFactor) inf2.causalQuery()
                .setTarget(target).setIntervention(intervention).run();




        System.out.println(exactRes);
        //System.out.println(inf.getModel().getFactor(5));


        // Approximate EM-based methods
        for(int i=0; i<numberPoints; i++) {

            // randomize P(U)
            StructuralCausalModel rmodel = (StructuralCausalModel) BayesianFactor.randomModel(causalModel,
                    5, false
                    ,causalModel.getExogenousVars()
            );

            // Run EM in the causal model
            ExpectationMaximization em =
                    new ExpectationMaximization(rmodel)
                            .setVerbose(false)
                            .setRegularization(0.0)
                            .setRecordIntermediate(false)
                            .setVerbose(true)
                            .setStopAtConvergence(true)
                            .setKlthreshold(0.00001)
                            //.setKlthreshold(0.0000001)
                            .setTrainableVars(causalModel.getExogenousVars());


            // run the method
            long t = System.currentTimeMillis();
            em.run(data, numberEMiter);


            if (em.getPerformedIterations()<numberEMiter) {
                EMtime[i] = System.currentTimeMillis() - t;

                //em.getIntermediateModels().stream().forEach(m->System.out.println(m.getFactor(5)));

                // Extract the learnt model
                StructuralCausalModel postModel = (StructuralCausalModel) em.getPosterior();
                System.out.println(postModel);

                bnets[i] = postModel.toBnet();

                //postModel.getEmpiricalNet().logProb(data); // problem with zeros
                // Run the  query
                t = System.nanoTime();
                CausalVE ve = new CausalVE(postModel);
                ifactors[i] = new BayesianToInterval().apply(ve.doQuery(target, intervention), target);
                UQtime[i] = (System.nanoTime() - t) / 1000000.0;
                EMiter[i] = em.getPerformedIterations();
                System.out.println(ifactors[i]);

            }else{
                System.out.println("non converging");
                i = i-1;
            }

        }

    }




    public static void parseArgs(String[] args){

        if (args.length > 0) {

            Options options = getArgOptions();
            CommandLineParser parser = new DefaultParser();
            HelpFormatter formatter = new HelpFormatter();

            CommandLine cmd = null;

            try {
                cmd = parser.parse(options, args);
            } catch (ParseException e) {
                System.out.println(e.getMessage());
                formatter.printHelp("utility-name", options);

                System.exit(1);
            }

            modelName = cmd.getOptionValue("model");
            numberPoints = Integer.parseInt(cmd.getOptionValue("numberPoints"));
            if(cmd.hasOption("numberEMiter")) numberEMiter = Integer.parseInt(cmd.getOptionValue("numberEMiter"));
            if(cmd.hasOption("samples")) samples = Integer.parseInt(cmd.getOptionValue("samples"));
            filterNonConverging = cmd.hasOption("filterNonConverging");

        }

    }


    public static Options getArgOptions(){

                        /*
                    --numberPoints 20 --numberEMiter 10 --samples 2000 --model ./models/poly4_12181.uai
                    -N 20 -n 10 -s 2000 -m ./models/poly4_12181.uai
                */
        Options options = new Options();

        options.addOption(Option.builder("N").longOpt("numberPoints").hasArg(true).required(true).build());
        options.addOption(Option.builder("n").longOpt("numberEMiter").hasArg(true).required(false).build());
        options.addOption(Option.builder("s").longOpt("samples").hasArg(true).required(false).build());
        options.addOption(Option.builder("m").longOpt("model").hasArg(true).required().build());
        options.addOption(Option.builder("f").longOpt("filterNonConverging").hasArg(false).required(false).build());


        return options;

    }


    public static void buildResults() throws InterruptedException {
        long t;

        // true result
        IntervalFactor exactInterval = new VertexToInterval().apply(exactRes, target);
        Map exactRes_ = intervalToDict("Ptrue", exactInterval);
        Map ALPres_ = intervalToDict("Paplp", ALPres);

        //EM-based results
        for(int i=1; i<=numberPoints; i++){

            //Store the numberOfPoints
            results.get(i).put("num_points", (double)i);

            //Store the numberOfPoints
            results.get(i).put("em_iter", (double) EMiter[i-1]);

            // Store the exact result
            results.get(i).putAll(exactRes_);

            // Store the ALP result
            results.get(i).putAll(ALPres_);

            // causal query is inside
            boolean inside = exactInterval.isInside(new BayesianFactor(ifactors[i-1].getDomain(), Doubles.concat(ifactors[i-1].getDataLower())));
            if(inside) results.get(i).put("inside", 1.0);
            else results.get(i).put("inside", 0.0);

            results.get(i).put("precise0", ifactors[i-1].getDataLower()[0][0]);
            results.get(i).put("precise1", ifactors[i-1].getDataLower()[0][1]);


            // get the converging i
            IntStream I = IntStream.range(0, i).filter(k -> !filterNonConverging || EMiter[k] < numberEMiter);

            IntervalFactor puq, pcn;

            double[][] values = new double[][]{IntStream.range(0, causalModel.getSize(target)).mapToDouble(k -> -1).toArray()};
            puq = new IntervalFactor(causalModel.getDomain(target), Strides.empty(), values, values);
            pcn = new IntervalFactor(causalModel.getDomain(target), Strides.empty(), values, values);



            if(I.count()>0) {
                // Queries union

                t = System.nanoTime();
                puq =
                        IntervalFactor.mergeBounds(
                                IntStream.range(0, i).mapToObj(k -> ifactors[k]).toArray(IntervalFactor[]::new)
                        );

                long t2 =  System.nanoTime();
                UQtime[i-1] += (t2 - t)/ 1000000.0;


                // Credal network
                t = System.nanoTime();

                SparseModel composed = VertexFactor.buildModel(true,
                        IntStream.range(0, i).mapToObj(k -> bnets[k]).toArray(BayesianNetwork[]::new)
                );

                CredalCausalVE credalVE = new CredalCausalVE(composed);
                VertexFactor res = (VertexFactor) credalVE.causalQuery().setIntervention(intervention).setTarget(target).run();
                pcn = new VertexToInterval().apply(res, target);
                CNtime[i-1] += (System.nanoTime() - t)/ 1000000.0;

            }

            results.get(i).putAll(intervalToDict("Puq", puq));
            results.get(i).putAll(intervalToDict("Pcn", pcn));

            results.get(i).put("em_time",  EMtime[i-1]);
            results.get(i).put("uq_time",  UQtime[i-1]);
            results.get(i).put("cn_time",  CNtime[i-1]);


        }
    }


    public static Map<String, Double> intervalToDict(String label, IntervalFactor f){
        Map<String, Double> out = new HashMap<String, Double>();
        double[] lbounds = Doubles.concat(f.getDataLower());
        double[] ubounds = Doubles.concat(f.getDataUpper());
        for(int i=0; i<lbounds.length; i++) {
            out.put(label + i + "_lbound", lbounds[i]);
        }
        for(int i=0; i<ubounds.length; i++){
            out.put(label+i+"_ubound", ubounds[i]);
        }
        return out;
    }


    public static void printResults(){
        System.out.println("<output>\n[");
        for(int k: results.keySet()){
            System.out.println(dictToString(results.get(k))+",");
        }
        System.out.println("]\n</output>");
    }

    public static String dictToString(Map<String, Double> d){
        String out = "{";
        List<String> keys = new ArrayList<String>(d.keySet());
        sort(keys);
        for(String k: keys){
            out+="'"+k+"': "+d.get(k)+",";
        }
        out = out.substring(0,out.length());
        out +="}";
        return out;

    }



}
